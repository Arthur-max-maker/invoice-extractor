# -*- coding: utf-8 -*-
"""
发票信息智能提取器 - Streamlit 应用
支持 PaddleOCR（本地）和百度智能云 OCR API（云端）双引擎自动切换
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
import requests

# ============================================================
# 自定义 CSS 样式
# ============================================================
CUSTOM_CSS = """
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }

    /* 功能卡片样式 */
    .info-card {
        background-color: #f5f5f5;
        border-left: 4px solid #1f77b4;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }

    /* 引擎状态标签 */
    .engine-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .engine-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .engine-inactive {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    /* 上传区域美化 */
    [data-testid="stFileUploader"] > div > div {
        border: 2px dashed #1f77b4 !important;
        border-radius: 10px !important;
        padding: 2rem !important;
    }

    /* 按钮美化 */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #155a8a;
        color: white;
    }

    /* 数据表格美化 */
    .dataframe-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

# ============================================================
# OCR 引擎检测与初始化
# ============================================================

# 检测 PaddleOCR 是否可用
PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


@st.cache_resource
def init_paddle_ocr() -> Any:
    """
    初始化 PaddleOCR 引擎（使用缓存避免重复加载模型）
    """
    if not PADDLE_AVAILABLE:
        return None
    try:
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
        )
        return ocr_engine
    except Exception as e:
        st.error(f"PaddleOCR 初始化失败：{e}")
        return None


def get_baidu_access_token(api_key: str, secret_key: str) -> Optional[str]:
    """
    通过百度智能云获取 access_token
    """
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    try:
        response = requests.post(token_url, params=params, timeout=10)
        result = response.json()
        if "access_token" in result:
            return result["access_token"]
        else:
            st.error(f"获取百度 access_token 失败：{result.get('error_description', '未知错误')}")
            return None
    except Exception as e:
        st.error(f"百度 API 请求异常：{e}")
        return None


def call_baidu_vat_ocr(image_bytes: bytes, access_token: str) -> Optional[Dict[str, Any]]:
    """
    调用百度智能云增值税发票 OCR API
    """
    api_url = (
        "https://aip.baidubce.com/rest/2.0/ocr/v1/vat_invoice"
        f"?access_token={access_token}"
    )
    # 将图片编码为 base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": encoded_image}

    try:
        response = requests.post(api_url, headers=headers, data=data, timeout=30)
        result = response.json()
        if "words_result" in result:
            return result
        else:
            st.warning(f"百度 OCR 返回异常：{result.get('error_msg', '未知错误')}")
            return None
    except Exception as e:
        st.error(f"百度 OCR API 调用异常：{e}")
        return None


# ============================================================
# 发票字段正则提取（用于 PaddleOCR 引擎）
# ============================================================

# 正则表达式规则列表：(字段名, 正则模式)
REGEX_RULES: List[Tuple[str, str]] = [
    ("发票代码", r'发票代码[：:]\s*(\d{10,12})'),
    ("发票号码", r'发票号码[：:]\s*(\d{8})'),
    ("开票日期", r'开票日期[：:]\s*(\d{4}年\d{2}月\d{2}日)'),
    ("金额", r'(?:不含税)?金额[：:]\s*[¥￥]?\s*([\d,]+\.?\d*)'),
    ("税额", r'(?:合计)?税额[：:]\s*[¥￥]?\s*([\d,]+\.?\d*)'),
    ("价税合计", r'价税合计[：:]\s*[¥￥]?\s*([\d,]+\.?\d*)'),
    ("价税合计", r'（小写）[）)]\s*[¥￥]?\s*([\d,]+\.?\d*)'),
    ("购买方名称", r'购买方[名称称][：:]\s*(.+)'),
    ("销售方名称", r'销售方[名称称][：:]\s*(.+)'),
]


def extract_invoice_fields_by_regex(ocr_text: str) -> Dict[str, str]:
    """
    使用正则表达式从 OCR 识别文本中提取发票关键字段
    """
    fields: Dict[str, str] = {
        "发票代码": "",
        "发票号码": "",
        "开票日期": "",
        "金额（不含税）": "",
        "税额": "",
        "价税合计": "",
        "购买方名称": "",
        "销售方名称": "",
    }

    # 将所有 OCR 文本合并为一个字符串用于匹配
    full_text = ocr_text

    for field_name, pattern in REGEX_RULES:
        match = re.search(pattern, full_text)
        if match:
            value = match.group(1).strip()
            # 如果该字段已有值且当前匹配也有效，优先保留更完整的匹配
            if field_name == "金额" and not fields["金额（不含税）"]:
                fields["金额（不含税）"] = value
            elif field_name == "价税合计" and not fields["价税合计"]:
                fields["价税合计"] = value
            elif field_name not in fields or not fields.get(field_name):
                fields[field_name] = value

    return fields


# ============================================================
# 发票处理核心逻辑
# ============================================================

def process_invoice_paddle(image_path_or_bytes: Any) -> Tuple[Dict[str, str], str]:
    """
    使用 PaddleOCR 引擎处理单张发票
    返回：(提取字段字典, 原始OCR文本)
    """
    ocr_engine = init_paddle_ocr()
    if ocr_engine is None:
        raise RuntimeError("PaddleOCR 引擎未初始化")

    # 执行 OCR 识别
    result = ocr_engine.ocr(image_path_or_bytes, cls=True)

    # 提取所有识别到的文字
    ocr_lines: List[str] = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]  # PaddleOCR 返回格式: [坐标, (文字, 置信度)]
                ocr_lines.append(text)

    full_text = "\n".join(ocr_lines)

    # 用正则提取字段
    fields = extract_invoice_fields_by_regex(full_text)

    return fields, full_text


def process_invoice_baidu(image_bytes: bytes, access_token: str) -> Tuple[Dict[str, str], str]:
    """
    使用百度智能云 OCR API 处理单张发票
    返回：(提取字段字典, 原始OCR文本)
    """
    api_result = call_baidu_vat_ocr(image_bytes, access_token)
    if api_result is None:
        raise RuntimeError("百度 OCR API 调用失败")

    words_result = api_result.get("words_result", {})

    # 百度 API 返回的结构化字段映射
    field_mapping = {
        "InvoiceCode": "发票代码",
        "InvoiceNum": "发票号码",
        "InvoiceDate": "开票日期",
        "AmountInFiguers": "金额（不含税）",
        "TaxAmount": "税额",
        "AmountInWords": "价税合计",
        "PurchaserName": "购买方名称",
        "SellerName": "销售方名称",
    }

    fields: Dict[str, str] = {
        "发票代码": "",
        "发票号码": "",
        "开票日期": "",
        "金额（不含税）": "",
        "税额": "",
        "价税合计": "",
        "购买方名称": "",
        "销售方名称": "",
    }

    # 拼接原始文本用于展示
    ocr_text_lines: List[str] = []

    for api_key, field_name in field_mapping.items():
        if api_key in words_result:
            word_info = words_result[api_key]
            if isinstance(word_info, dict) and "word" in word_info:
                fields[field_name] = word_info["word"].strip()
                ocr_text_lines.append(f"{field_name}: {word_info['word']}")
            elif isinstance(word_info, str):
                fields[field_name] = word_info.strip()
                ocr_text_lines.append(f"{field_name}: {word_info}")

    # 补充：价税合计优先取小写金额
    if "TotalAmount" in words_result:
        total_info = words_result["TotalAmount"]
        if isinstance(total_info, dict) and "word" in total_info:
            if not fields["价税合计"]:
                fields["价税合计"] = total_info["word"].strip()

    full_text = "\n".join(ocr_text_lines) if ocr_text_lines else str(words_result)

    return fields, full_text


def process_single_invoice(
    image_file,
    engine: str,
    baidu_access_token: Optional[str] = None,
) -> Tuple[Dict[str, str], str, str]:
    """
    处理单张发票的统一入口
    参数：
        image_file: Streamlit 上传的文件对象
        engine: "paddle" 或 "baidu"
        baidu_access_token: 百度 API 的 access_token（引擎2需要）
    返回：
        (字段字典, 原始OCR文本, 文件名)
    """
    filename = image_file.name

    if engine == "paddle":
        # PaddleOCR 支持直接传入文件对象
        fields, ocr_text = process_invoice_paddle(image_file)
    elif engine == "baidu":
        image_bytes = image_file.read()
        fields, ocr_text = process_invoice_baidu(image_bytes, baidu_access_token)
    else:
        raise ValueError(f"不支持的引擎类型：{engine}")

    return fields, ocr_text, filename


# ============================================================
# 工具函数
# ============================================================

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """将 PIL Image 转换为字节数据"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def validate_image_file(filename: str) -> bool:
    """验证文件是否为支持的图片格式"""
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


# ============================================================
# Streamlit 应用主体
# ============================================================

def main() -> None:
    # 注入自定义 CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 页面标题
    st.markdown('<div class="main-title">🧾 发票信息智能提取器</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------
    # 侧边栏配置
    # ----------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 引擎配置")

        # 引擎状态显示
        st.markdown("### 🔍 引擎状态")
        if PADDLE_AVAILABLE:
            st.markdown(
                '<span class="engine-badge engine-active">✅ PaddleOCR（本地）可用</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="engine-badge engine-inactive">❌ PaddleOCR 未安装</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 百度 API 密钥输入（可折叠）
        with st.expander("🔑 百度智能云 API 配置", expanded=False):
            st.markdown(
                '<div class="info-card">'
                '如未安装 PaddleOCR，可使用百度智能云 OCR API 作为备选方案。<br>'
                '请前往 <a href="https://cloud.baidu.com/product/ocr" target="_blank">'
                '百度智能云控制台</a> 创建应用获取 API Key 和 Secret Key。'
                '</div>',
                unsafe_allow_html=True,
            )
            baidu_api_key = st.text_input("API Key", type="password", key="baidu_api_key")
            baidu_secret_key = st.text_input("Secret Key", type="password", key="baidu_secret_key")

        st.markdown("---")

        # 使用说明
        with st.expander("📖 使用说明", expanded=False):
            st.markdown(
                '<div class="info-card">'
                '📋 <b>操作步骤：</b><br>'
                '1. 上传发票图片（支持 jpg/png/bmp，可多张批量上传）<br>'
                '2. 点击「开始识别」按钮<br>'
                '3. 查看识别结果表格<br>'
                '4. 可导出为 Excel 文件<br><br>'
                '🔧 <b>引擎优先级：</b><br>'
                '• 优先使用 PaddleOCR（本地识别，无需网络）<br>'
                '• 若 PaddleOCR 不可用，自动切换到百度智能云 API<br>'
                '• 若均不可用，请在侧边栏安装 PaddleOCR 或配置百度 API 密钥'
                '</div>',
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------------
    # 确定当前使用的引擎
    # ----------------------------------------------------------
    current_engine: Optional[str] = None
    baidu_access_token: Optional[str] = None

    if PADDLE_AVAILABLE:
        current_engine = "paddle"
    elif baidu_api_key and baidu_secret_key:
        # 尝试获取百度 access_token
        baidu_access_token = get_baidu_access_token(baidu_api_key, baidu_secret_key)
        if baidu_access_token:
            current_engine = "baidu"
        else:
            current_engine = None
    else:
        current_engine = None

    # ----------------------------------------------------------
    # 主区域 - 文件上传
    # ----------------------------------------------------------
    st.markdown("### 📁 上传发票图片")
    st.markdown(
        '<div class="info-card">'
        '支持格式：JPG / PNG / BMP，可同时上传多张发票图片进行批量识别'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="选择发票图片",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="invoice_uploader",
        label_visibility="collapsed",
    )

    # ----------------------------------------------------------
    # 引擎不可用提示
    # ----------------------------------------------------------
    if current_engine is None:
        st.error(
            "⚠️ 当前没有可用的 OCR 引擎！\n\n"
            "请选择以下方式之一：\n"
            "1. 安装 PaddleOCR：`pip install paddlepaddle paddleocr opencv-python-headless`\n"
            "2. 在侧边栏填写百度智能云 API Key 和 Secret Key"
        )
        st.stop()

    # 显示当前引擎信息
    engine_label = "PaddleOCR（本地）" if current_engine == "paddle" else "百度智能云 OCR API（云端）"
    st.info(f"📌 当前使用引擎：**{engine_label}**")

    # ----------------------------------------------------------
    # 开始识别按钮
    # ----------------------------------------------------------
    if uploaded_files:
        count = len(uploaded_files)
        st.markdown(f"已选择 **{count}** 张图片")

        # 预览上传的图片（最多显示 12 张缩略图，超出折叠）
        preview_files = uploaded_files[:12]
        with st.expander(f"📷 图片预览（{count} 张）", expanded=(count <= 6)):
            cols = st.columns(min(len(preview_files), 4))
            for idx, file in enumerate(preview_files):
                with cols[idx % 4]:
                    try:
                        img = Image.open(file)
                        st.image(img, caption=file.name, use_container_width=True)
                    except Exception:
                        st.warning(f"无法预览：{file.name}")
            if count > 12:
                st.caption(f"…还有 {count - 12} 张未显示")

        st.markdown("---")
        start_btn = st.button(
            f"🚀 开始识别（{count} 张）",
            type="primary",
            use_container_width=True,
        )

        if start_btn:
            process_invoices(uploaded_files, current_engine, baidu_access_token)
    else:
        st.info("👆 请先上传发票图片，然后点击「开始识别」按钮")


def process_invoices(
    uploaded_files: List[Any],
    engine: str,
    baidu_access_token: Optional[str],
) -> None:
    """
    批量处理上传的发票图片
    """
    # 存储所有结果
    all_results: List[Dict[str, str]] = []
    all_ocr_texts: List[Tuple[str, str]] = []  # (文件名, OCR文本)

    # 定义结果列
    columns = [
        "文件名",
        "发票代码",
        "发票号码",
        "开票日期",
        "金额（不含税）",
        "税额",
        "价税合计",
        "购买方名称",
        "销售方名称",
    ]

    # 逐张处理
    progress_bar = st.progress(0, text="准备中...")

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        progress_text = f"正在处理第 {i + 1}/{len(uploaded_files)} 张：{filename}"
        progress_bar.progress((i) / len(uploaded_files), text=progress_text)

        try:
            with st.spinner(f"🔍 正在识别：{filename}"):
                fields, ocr_text, fname = process_single_invoice(
                    uploaded_file, engine, baidu_access_token
                )

                # 构建结果行
                row: Dict[str, str] = {"文件名": fname}
                for col in columns[1:]:
                    row[col] = fields.get(col, "")

                all_results.append(row)
                all_ocr_texts.append((fname, ocr_text))

        except Exception as e:
            st.error(f"处理文件 {filename} 时出错：{e}")
            # 添加空行保持表格对齐
            row = {"文件名": filename}
            for col in columns[1:]:
                row[col] = f"识别失败: {str(e)}"
            all_results.append(row)
            all_ocr_texts.append((filename, f"识别失败: {str(e)}"))

        # 重置文件指针，以便后续可能的重新读取
        uploaded_file.seek(0)

    progress_bar.progress(1.0, text="识别完成！")

    # ----------------------------------------------------------
    # 显示结果（每张发票：图片 + 可编辑表格）
    # ----------------------------------------------------------
    if all_results:
        st.markdown("---")
        st.markdown(f"### 📊 识别结果（共 {len(all_results)} 张，可直接点击单元格修正）")

        # 逐张显示：原始图片 + 可编辑字段（多张时折叠）
        detail_expander = st.expander(
            f"🔍 逐张核对（{len(all_results)} 张）",
            expanded=(len(all_results) <= 3),
        )
        with detail_expander:
            for idx, (uploaded_file, result_row) in enumerate(zip(uploaded_files, all_results)):
                with st.container():
                    st.markdown(f"#### 🧾 {result_row['文件名']}")

                    # 两列布局：左图右表
                    img_col, data_col = st.columns([1, 2])

                    with img_col:
                        # 显示原始发票图片
                        try:
                            uploaded_file.seek(0)
                            img = Image.open(uploaded_file)
                            st.image(
                                img,
                                caption="原始发票图片（点击核对）",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"无法加载图片：{e}")

                    with data_col:
                        # 将单条结果转为可编辑 DataFrame
                        single_df = pd.DataFrame([result_row], columns=columns)
                        edited_df = st.data_editor(
                            single_df,
                            use_container_width=True,
                            hide_index=True,
                            num_rows="fixed",
                            disabled=["文件名"],
                            key=f"editor_single_{idx}",
                            column_config={
                                "文件名": st.column_config.TextColumn("文件名", width="medium", disabled=True),
                                "发票代码": st.column_config.TextColumn("发票代码", width="medium"),
                                "发票号码": st.column_config.TextColumn("发票号码", width="medium"),
                                "开票日期": st.column_config.TextColumn("开票日期", width="medium"),
                                "金额（不含税）": st.column_config.TextColumn("金额（不含税）", width="medium"),
                                "税额": st.column_config.TextColumn("税额", width="medium"),
                                "价税合计": st.column_config.TextColumn("价税合计", width="medium"),
                                "购买方名称": st.column_config.TextColumn("购买方名称", width="large"),
                                "销售方名称": st.column_config.TextColumn("销售方名称", width="large"),
                            },
                        )
                        # 用编辑后的数据更新结果
                        if not edited_df.empty:
                            all_results[idx] = edited_df.iloc[0].to_dict()

                    st.markdown("---")

        # ----------------------------------------------------------
        # 汇总可编辑表格
        # ----------------------------------------------------------
        st.markdown("### 📋 全部结果汇总（可直接修正后导出）")
        st.caption("💡 点击任意单元格即可编辑，修改后的数据将用于最终导出。")

        df = pd.DataFrame(all_results, columns=columns)

        edited_total_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["文件名"],
            key="editor_total",
            column_config={
                "文件名": st.column_config.TextColumn("文件名", width="medium", disabled=True),
                "发票代码": st.column_config.TextColumn("发票代码", width="medium"),
                "发票号码": st.column_config.TextColumn("发票号码", width="medium"),
                "开票日期": st.column_config.TextColumn("开票日期", width="medium"),
                "金额（不含税）": st.column_config.TextColumn("金额（不含税）", width="medium"),
                "税额": st.column_config.TextColumn("税额", width="medium"),
                "价税合计": st.column_config.TextColumn("价税合计", width="medium"),
                "购买方名称": st.column_config.TextColumn("购买方名称", width="large"),
                "销售方名称": st.column_config.TextColumn("销售方名称", width="large"),
            },
        )

        # ----------------------------------------------------------
        # 导出 Excel（使用编辑后的数据）
        # ----------------------------------------------------------
        st.markdown("---")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            edited_total_df.to_excel(writer, index=False, sheet_name="发票识别结果")

            # 调整列宽
            worksheet = writer.sheets["发票识别结果"]
            for col_idx, col_name in enumerate(columns, 1):
                max_len = max(
                    edited_total_df[col_name].astype(str).map(len).max() if len(edited_total_df) > 0 else 0,
                    len(col_name),
                )
                # 中文字符宽度约为英文的2倍
                adjusted_width = min(max_len * 2 + 2, 50)
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=col_idx).column_letter
                ].width = adjusted_width

        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="📥 导出修正后的 Excel",
            data=excel_data,
            file_name=f"发票识别结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # ----------------------------------------------------------
        # 显示每张发票的原始 OCR 文本（可折叠）
        # ----------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔎 原始 OCR 文本")

        for fname, ocr_text in all_ocr_texts:
            with st.expander(f"📄 {fname}", expanded=False):
                st.text_area(
                    label="OCR识别文本",
                    value=ocr_text,
                    height=200,
                    key=f"ocr_text_{fname}_{id(ocr_text)}",
                    label_visibility="collapsed",
                )

        # 成功提示
        success_count = sum(
            1 for r in all_results if not any(
                str(v).startswith("识别失败") for v in r.values()
            )
        )
        st.success(
            f"✅ 处理完成！共 {len(all_results)} 张发票，"
            f"成功识别 {success_count} 张。"
        )
    else:
        st.warning("未识别到任何发票信息，请检查图片质量后重试。")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    main()

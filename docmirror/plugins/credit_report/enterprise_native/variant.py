# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for native-text enterprise credit reports."""

from copy import deepcopy
from typing import Any

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter

_ENTERPRISE_DOCUMENT_DATASET_ORDER = (
    "enterprise_report_metadata",
    "report_notes",
    "enterprise_exchange_rates",
    "enterprise_report_identity",
    "enterprise_section_presence",
    "enterprise_dispute_overview",
    "enterprise_credit_overview",
    "enterprise_public_record_counts",
    "enterprise_current_credit_summary",
    "enterprise_facility_summary",
    "enterprise_repayment_responsibility_summary",
    "enterprise_closed_credit_summary",
    "enterprise_recovery_summary",
    "enterprise_overdue_summary",
    "enterprise_profile",
    "enterprise_capital_summary",
    "enterprise_contributors",
    "enterprise_key_personnel",
    "enterprise_relationships",
    "enterprise_credit_detail_groups",
    "enterprise_displayed_credit_summary",
    "enterprise_credit_accounts",
    "enterprise_account_annotations",
    "enterprise_interest_arrears",
    "enterprise_credit_facilities",
    "enterprise_repayment_responsibility_group_details",
    "enterprise_repayment_responsibility_accounts",
    "enterprise_public_utility_payment_records",
    "enterprise_public_tax_arrears_records",
    "enterprise_public_civil_judgment_records",
    "enterprise_public_enforcement_records",
    "enterprise_public_administrative_penalty_records",
    "enterprise_public_housing_fund_payment_records",
    "enterprise_public_social_security_payment_records",
    "enterprise_public_license_records",
    "enterprise_public_certification_records",
    "enterprise_public_qualification_records",
    "enterprise_public_award_records",
    "enterprise_public_export_quality_records",
    "enterprise_public_inspection_exemption_records",
    "enterprise_public_regulatory_supervision_records",
    "enterprise_public_patent_records",
    "enterprise_public_financing_restriction_records",
    "enterprise_public_data_provider_statement_records",
    "enterprise_public_credit_bureau_statement_records",
    "enterprise_public_subject_statement_records",
    "enterprise_public_dispute_annotation_records",
    "enterprise_attachment_accounts",
    "enterprise_credit_supplement",
    "enterprise_attachment_credit_details",
    "enterprise_special_transactions",
    "enterprise_utility_payment_history",
    "enterprise_housing_fund_history",
)


class EnterpriseNativeVariant(CreditReportVariantAdapter):
    """Keep enterprise extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="enterprise_native",
            report_subtype="enterprise",
            expected_content_modes=frozenset({"native_text", "mixed"}),
            include_credit_lines=True,
        )

    def dataset_names(self) -> tuple[str, ...]:
        """Publish public records through typed business tables, not content rows."""
        return _ENTERPRISE_DOCUMENT_DATASET_ORDER

    def data_dictionary(self) -> dict[str, Any]:
        """Return an enterprise-owned dictionary with no personal inheritance."""
        from docmirror.plugins.credit_report.enterprise_native.schema import (
            enterprise_credit_report_data_dictionary,
        )

        dictionary = enterprise_credit_report_data_dictionary()
        fields = dictionary.setdefault("fields", {})
        fields["subject_name"] = {"label": "企业名称", "type": "string"}
        fields["unified_social_credit_code"] = {
            "label": "统一社会信用代码",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["zhongzheng_code"] = {
            "label": "中征码",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["organization_code"] = {
            "label": "组织机构代码",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["institution_credit_code"] = {
            "label": "机构信用代码",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["national_tax_id"] = {
            "label": "纳税人识别号（国税）",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["local_tax_id"] = {
            "label": "纳税人识别号（地税）",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["business_registration_number"] = {
            "label": "工商注册号",
            "type": "string",
            "format": "long_id",
            "sensitive": True,
        }
        fields["query_institution"] = {"label": "查询机构", "type": "string"}
        fields.update(
            {
                "report_number": {
                    "label": "报告编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "projected_account_count": {"label": "投影账户数", "type": "integer"},
                "page": {"label": "页码", "type": "integer"},
                "enterprise_identity_id": {"label": "企业报告身份记录ID", "type": "string"},
                "enterprise_name": {"label": "企业名称", "type": "string"},
                "report_time": {"label": "报告时间", "type": "datetime"},
                "source_page_end": {"label": "源结束页码", "type": "integer"},
                "enterprise_dispute_overview_id": {"label": "异议概要ID", "type": "string"},
                "in_progress_dispute_count": {"label": "处理中异议笔数", "type": "integer"},
                "dispute_status": {
                    "label": "异议状态",
                    "type": "string",
                    "enum_ref": "dispute_status",
                },
                "enterprise_credit_overview_id": {"label": "信用概要ID", "type": "string"},
                "enterprise_public_record_count_id": {
                    "label": "公共记录计数ID",
                    "type": "string",
                },
                "record_count": {"label": "记录条数", "type": "integer"},
                "enterprise_recovery_summary_id": {"label": "被追偿概要ID", "type": "string"},
                "recovery_type": {"label": "被追偿业务类型", "type": "string"},
                "account_count": {"label": "账户数", "type": "integer"},
                "latest_disposal_date": {"label": "最近一次处置日期", "type": "date"},
                "latest_repayment_date": {"label": "最近一次还款日期", "type": "date"},
                "enterprise_overdue_summary_id": {"label": "逾期概要ID", "type": "string"},
                "overdue_interest_and_other": {"label": "逾期利息及其他", "type": "money"},
                "enterprise_profile_id": {"label": "企业基本信息ID", "type": "string"},
                "economic_type": {"label": "经济类型", "type": "string"},
                "organization_type": {"label": "组织机构类型", "type": "string"},
                "enterprise_scale": {"label": "企业规模", "type": "string"},
                "industry": {"label": "所属行业", "type": "string"},
                "establishment_year": {"label": "成立年份", "type": "integer"},
                "registration_certificate_valid_through": {
                    "label": "登记证书有效截止日期",
                    "type": "date",
                },
                "registered_address": {"label": "登记地址", "type": "string"},
                "operating_address": {"label": "办公/经营地址", "type": "string"},
                "operating_status": {"label": "存续状态", "type": "string"},
                "source_state": {
                    "label": "源报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "section_presence_id": {"label": "规范章节状态ID", "type": "string"},
                "section_key": {"label": "规范章节键", "type": "string"},
                "section_title": {"label": "规范章节名称", "type": "string"},
                "presence_status": {
                    "label": "章节存在状态",
                    "type": "string",
                    "enum_ref": "section_presence_status",
                },
                "heading_detected": {"label": "是否识别到章节标题", "type": "boolean"},
                "count_scope": {"label": "计数口径", "type": "string"},
                "interest_arrears_id": {"label": "欠息记录ID", "type": "string"},
                "arrears_type": {"label": "欠息类型", "type": "string"},
                "arrears_balance": {"label": "欠息余额", "type": "money"},
                "balance_change_date": {"label": "余额变化日期", "type": "date"},
                "amount_due": {"label": "本月应缴金额", "type": "money"},
                "amount_paid": {"label": "本月实缴金额", "type": "money"},
                "account_type": {"label": "账户类型", "type": "string"},
                "maturity_date": {
                    "label": "到期日期",
                    "type": "date",
                    "definition": "合同或授信协议的到期日；不是许可或证书有效期。",
                },
                "credit_limit": {"label": "信用额度", "type": "money"},
                "loan_amount": {"label": "借款金额", "type": "money"},
                "discount_amount": {"label": "贴现金额", "type": "money"},
                "instrument_amount": {"label": "票据/信用证金额", "type": "money"},
                "guarantee_amount": {"label": "担保金额", "type": "money"},
                "risk_exposure_amount": {"label": "风险敞口", "type": "money"},
                "deposit_ratio": {"label": "保证金比例", "type": "percentage"},
                "credit_limit_status": {
                    "label": "信用额度报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "loan_amount_status": {
                    "label": "借款金额报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "discount_amount_status": {
                    "label": "贴现金额报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "instrument_amount_status": {
                    "label": "票据/信用证金额报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "guarantee_amount_status": {
                    "label": "担保金额报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "balance_status": {"label": "余额报告状态", "type": "string"},
                "credit_balance": {"label": "借贷交易余额", "type": "money"},
                "guarantee_balance": {"label": "担保交易余额", "type": "money"},
                "recovered_debt_balance": {"label": "被追偿余额", "type": "money"},
                "first_credit_year": {"label": "首次有信贷交易的年份", "type": "integer"},
                "first_credit_year_status": {
                    "label": "首次信贷交易年份报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "credit_institution_count": {"label": "发生信贷交易的机构数", "type": "integer"},
                "active_credit_institution_count": {
                    "label": "当前有未结清信贷交易的机构数",
                    "type": "integer",
                },
                "first_repayment_responsibility_year": {
                    "label": "首次有相关还款责任的年份",
                    "type": "integer",
                },
                "first_repayment_responsibility_year_status": {
                    "label": "首次相关还款责任年份报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "credit_attention_balance": {"label": "借贷交易关注类余额", "type": "money"},
                "credit_adverse_balance": {"label": "借贷交易不良类余额", "type": "money"},
                "guarantee_attention_balance": {"label": "担保交易关注类余额", "type": "money"},
                "guarantee_adverse_balance": {"label": "担保交易不良类余额", "type": "money"},
                "amount_unit": {"label": "金额单位", "type": "string"},
                "reported_account_count": {"label": "源表账户数", "type": "integer"},
                "extracted_account_count": {"label": "提取账户数", "type": "integer"},
                "canonical_table_account_count": {
                    "label": "规范表账户数",
                    "type": "integer",
                },
                "credit_line_count": {"label": "授信额度记录数", "type": "integer"},
                "public_record_overview_counts": {
                    "label": "信息概要固定类别记录数",
                    "type": "object",
                    "map_key_enum": "record_type",
                },
                "public_record_overview_count_scope": {
                    "label": "信息概要记录计数口径",
                    "type": "string",
                },
                "extracted_public_record_count": {
                    "label": "公共记录明细提取行数",
                    "type": "integer",
                },
                "extracted_public_record_type_counts": {
                    "label": "公共记录明细类型统计",
                    "type": "object",
                    "map_key_enum": "record_type",
                },
                "extracted_public_record_count_scope": {
                    "label": "公共记录明细计数口径",
                    "type": "string",
                },
                "audit_id": {"label": "完整性审计ID", "type": "string"},
                "continuation_family": {
                    "label": "连续记录类型",
                    "type": "string",
                },
                "expected_record_count": {
                    "label": "源合同记录数",
                    "type": "integer",
                },
                "extracted_record_count": {
                    "label": "逻辑记录数",
                    "type": "integer",
                },
                "unresolved_record_count": {
                    "label": "未解析记录数",
                    "type": "integer",
                },
                "unexpected_record_count": {
                    "label": "超出源合同的记录数",
                    "type": "integer",
                },
                "reconciliation_status": {
                    "label": "核对状态",
                    "type": "string",
                },
                "reported_account_balance": {
                    "label": "源表账户余额合计",
                    "type": "money",
                },
                "reported_account_counts": {
                    "label": "各类账户数",
                    "type": "object",
                },
                "reported_account_balances": {
                    "label": "各类账户余额",
                    "type": "object",
                },
                "reported_credit_line_count": {
                    "label": "源表授信额度记录数",
                    "type": "integer",
                },
                "source_account_summary_table_id": {
                    "label": "源账户概要表标识",
                    "type": "string",
                },
                "source_account_summary_page": {
                    "label": "源账户概要表页码",
                    "type": "integer",
                },
                "facility_summary_record_count": {"label": "授信额度汇总类别数", "type": "integer"},
                "facility_summary": {"label": "授信额度汇总（兼容视图）", "type": "object"},
                "account_population_comparable": {
                    "label": "账户明细与概要口径可比",
                    "type": "boolean",
                },
                "account_dataset_scope": {"label": "信贷账户数据口径", "type": "string"},
                "account_dataset_scope_note": {
                    "label": "信贷账户及附件说明",
                    "type": "text",
                },
                "source_display_limited": {
                    "label": "源报告是否声明信息展示范围受限",
                    "type": "boolean",
                    "definition": (
                        "包括受篇幅限制仅展示部分信贷记录，或仅展示一定期限范围内的"
                        "已结清信贷、非信贷和公共信息。"
                    ),
                },
                "attachment_account_count": {
                    "label": "附件账户/业务数",
                    "type": "integer",
                },
                "attachment_history_row_count": {
                    "label": "附件逐期信用记录行数",
                    "type": "integer",
                },
                "attachment_detail_card_count": {
                    "label": "附件信贷明细卡片数",
                    "type": "integer",
                },
                "attachment_special_transaction_count": {
                    "label": "附件特定交易数",
                    "type": "integer",
                },
                "report_edition": {"label": "报告版本", "type": "string"},
                "exchange_rate_usd_cny": {
                    "label": "汇率（美元折人民币）",
                    "type": "number",
                },
                "exchange_rate_effective_period": {
                    "label": "汇率有效期",
                    "type": "string",
                },
                "field": {"label": "信息项", "type": "string"},
                "value": {"label": "信息值", "type": "string"},
                "source_institution": {"label": "信息来源机构", "type": "string"},
                "update_date": {"label": "更新日期", "type": "date"},
                "role": {"label": "角色/职位", "type": "string"},
                "name": {"label": "名称/姓名", "type": "string"},
                "identity_type": {"label": "身份标识类型", "type": "string"},
                "identity_number": {
                    "label": "身份标识号码",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "ownership_percentage": {"label": "出资比例", "type": "string"},
                "relationship_type": {"label": "关系类型", "type": "string"},
                "registered_capital_amount": {"label": "注册资本", "type": "money"},
                "contributor_count": {"label": "主要出资人记录数", "type": "integer"},
                "contributor_status": {"label": "主要出资人信息状态", "type": "string"},
                "sequence": {"label": "序号", "type": "integer"},
                "public_record_id": {"label": "公共记录ID", "type": "string"},
                "record_type": {"label": "记录类型", "type": "string"},
                "authority": {"label": "主管/发布机构", "type": "string"},
                "category": {"label": "记录类别", "type": "string"},
                "start_date": {"label": "生效/发生日期", "type": "date"},
                "end_date": {"label": "截止日期", "type": "date"},
                "content": {"label": "记录内容", "type": "string"},
                "details": {"label": "源字段明细", "type": "object"},
                "attributes": {"label": "类型化属性", "type": "object"},
                "statistics_month": {"label": "统计年月", "type": "string"},
                "initial_contribution_month": {"label": "初缴年月", "type": "string"},
                "employee_count": {"label": "职工人数", "type": "integer"},
                "contribution_base": {"label": "缴费/缴存基数", "type": "money"},
                "last_contribution_date": {"label": "最近一次缴费/缴存日期", "type": "date"},
                "paid_through_month": {"label": "缴至年月", "type": "string"},
                "payment_status": {"label": "缴费/缴存状态", "type": "string"},
                "cumulative_arrears": {"label": "累计欠费/欠缴金额", "type": "money"},
                "licensing_authority": {"label": "许可部门", "type": "string"},
                "license_type": {"label": "许可类型", "type": "string"},
                "license_date": {"label": "许可日期", "type": "date"},
                "license_expiry_date": {"label": "许可截止日期", "type": "date"},
                "license_content": {"label": "许可内容", "type": "string"},
                "certification_authority": {"label": "认证部门", "type": "string"},
                "certification_type": {"label": "认证类型", "type": "string"},
                "certification_date": {"label": "认证日期", "type": "date"},
                "certification_expiry_date": {"label": "认证截止日期", "type": "date"},
                "certification_content": {"label": "认证内容", "type": "string"},
                "source_page": {"label": "源页码", "type": "integer"},
                "contributor_source_page": {
                    "label": "主要出资人表源页码",
                    "type": "integer",
                },
                "source_table_id": {"label": "源表标识", "type": "string"},
                "source_row_number": {"label": "源表行号", "type": "integer"},
                "column_count": {"label": "列数", "type": "integer"},
                "row_text": {"label": "源表行内容", "type": "string"},
                "credit_line_id": {"label": "授信记录ID", "type": "string"},
                "facility_type": {"label": "授信类型", "type": "string"},
                "account_state": {"label": "账户开闭状态", "type": "string"},
                "activation_state": {"label": "激活状态", "type": "string"},
                "current_overdue": {"label": "当前是否逾期", "type": "boolean"},
                "payoff_state": {"label": "结清状态", "type": "string"},
                "total_limit": {"label": "总额度", "type": "money"},
                "total_limit_status": {"label": "总额度报告状态", "type": "string"},
                "used_limit": {"label": "已用额度", "type": "money"},
                "used_limit_status": {"label": "已用额度报告状态", "type": "string"},
                "available_limit": {"label": "剩余可用额度", "type": "money"},
                "available_limit_status": {
                    "label": "剩余可用额度报告状态",
                    "type": "string",
                },
                "currency": {"label": "币种", "type": "string"},
                "utilization_rate": {
                    "label": "额度使用率",
                    "type": "percentage",
                },
                "non_credit_accounts": {"label": "非信贷交易账户数", "type": "integer"},
                "tax_arrears": {"label": "欠税记录条数", "type": "integer"},
                "civil_judgments": {"label": "民事判决记录条数", "type": "integer"},
                "enforcements": {"label": "强制执行记录条数", "type": "integer"},
                "administrative_penalties": {"label": "行政处罚记录条数", "type": "integer"},
                "supplement_id": {"label": "补充记录ID", "type": "string"},
                "account_id": {"label": "账户记录ID", "type": "string"},
                "account_identifier": {
                    "label": "账户标识",
                    "type": "long_id",
                    "sensitive": True,
                },
                "contract_number": {
                    "label": "保证合同编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "responsibility_amount": {
                    "label": "还款责任金额",
                    "type": "money",
                },
                "snapshot_date": {"label": "信息截至日期", "type": "date"},
                "institution": {"label": "授信机构", "type": "string"},
                "business_type": {"label": "业务类型", "type": "string"},
                "business_category": {"label": "业务类别", "type": "string"},
                "report_date": {"label": "信息报告日期", "type": "date"},
                "open_date": {"label": "开立日期", "type": "date"},
                "due_date": {
                    "label": "原始兼容到期日",
                    "type": "date",
                    "deprecated": True,
                    "canonical_field": "maturity_date",
                },
                "close_date": {"label": "关闭日期", "type": "date"},
                "balance": {"label": "余额", "type": "money"},
                "five_tier_class": {"label": "五级分类", "type": "string"},
                "five_tier_class_source": {
                    "label": "五级分类来源",
                    "type": "string",
                },
                "classification_date": {"label": "五级分类认定日期", "type": "date"},
                "overdue_total": {"label": "逾期总额", "type": "money"},
                "overdue_principal": {
                    "label": "逾期本金",
                    "type": "money",
                    "definition": "源报告明确展示0时保留0；仅未报告时为空。",
                },
                "overdue_months": {"label": "逾期月数", "type": "integer"},
                "scheduled_repayment_date": {"label": "最近约定还款日期", "type": "date"},
                "scheduled_repayment_amount": {"label": "最近应还总额", "type": "money"},
                "actual_repayment_date": {"label": "最近实际还款日期", "type": "date"},
                "actual_repayment_amount": {"label": "最近实还总额", "type": "money"},
                "repayment_method": {"label": "最近还款形式", "type": "string"},
                "remaining_periods": {
                    "label": "剩余还款月数",
                    "type": "integer",
                    "definition": "源报告字段为剩余还款月数。",
                },
                "issuance_form": {"label": "发放形式", "type": "string"},
                "guarantee_type": {"label": "担保方式", "type": "string"},
                "counter_guarantee_type": {"label": "反担保方式", "type": "string"},
                "status": {"label": "账户状态", "type": "string"},
                "current_overdue_status": {
                    "label": "当前逾期报告状态",
                    "type": "string",
                },
                "last_repayment_date": {"label": "最近还款日期", "type": "date"},
                "special_transaction": {"label": "特定交易提示", "type": "string"},
                "credit_agreement_identifier": {
                    "label": "授信协议编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "credit_agreement_status": {
                    "label": "授信协议编号报告状态",
                    "type": "string",
                },
                "history_status": {"label": "历史表现", "type": "string"},
                "closed_summary_id": {"label": "已结清概要ID", "type": "string"},
                "current_summary_id": {"label": "当前信贷概要ID", "type": "string"},
                "displayed_summary_id": {"label": "明细分组汇总ID", "type": "string"},
                "transaction_group": {"label": "交易分组", "type": "string"},
                "settlement_status": {"label": "结清状态", "type": "string"},
                "normal_account_count": {"label": "正常类账户数", "type": "integer"},
                "normal_balance": {"label": "正常类余额", "type": "money"},
                "attention_account_count": {"label": "关注类账户数", "type": "integer"},
                "attention_balance": {"label": "关注类余额", "type": "money"},
                "adverse_account_count": {"label": "不良类账户数", "type": "integer"},
                "adverse_balance": {"label": "不良类余额", "type": "money"},
                "total_account_count": {"label": "账户合计", "type": "integer"},
                "total_balance": {"label": "余额合计", "type": "money"},
                "source_group_account_count": {"label": "源分组账户总数", "type": "integer"},
                "source_account_count": {"label": "源汇总行账户数", "type": "integer"},
                "source_reported_amount": {"label": "源报告金额", "type": "money"},
                "amount_kind": {
                    "label": "源报告金额口径",
                    "type": "string",
                    "enum_ref": "amount_kind",
                },
                "summary_scope": {"label": "汇总口径", "type": "string"},
                "is_total": {"label": "是否合计行", "type": "boolean"},
                "responsibility_summary_id": {
                    "label": "还款责任概要ID",
                    "type": "string",
                },
                "responsibility_type": {"label": "责任类型", "type": "string"},
                "recovered_responsibility_amount": {
                    "label": "被追偿业务还款责任金额",
                    "type": "money",
                },
                "recovered_account_count": {
                    "label": "被追偿业务账户数",
                    "type": "integer",
                },
                "recovered_balance": {"label": "被追偿业务余额", "type": "money"},
                "other_credit_responsibility_amount": {
                    "label": "其他借贷交易还款责任金额",
                    "type": "money",
                },
                "other_credit_account_count": {
                    "label": "其他借贷交易账户数",
                    "type": "integer",
                },
                "other_credit_balance": {
                    "label": "其他借贷交易余额",
                    "type": "money",
                },
                "other_credit_attention_balance": {
                    "label": "其他借贷交易关注类余额",
                    "type": "money",
                },
                "other_credit_adverse_balance": {
                    "label": "其他借贷交易不良类余额",
                    "type": "money",
                },
                "guarantee_responsibility_amount": {
                    "label": "担保交易还款责任金额",
                    "type": "money",
                },
                "guarantee_account_count": {
                    "label": "担保交易账户数",
                    "type": "integer",
                },
                "loan_or_credit_amount": {
                    "label": "借款金额/信用额度",
                    "type": "money",
                },
                "overdue_months_or_repayment_status": {
                    "label": "逾期月数/还款状态",
                    "type": "string",
                },
                "source_table_id_end": {
                    "label": "源结束表标识",
                    "type": "string",
                },
                "contract_number_status": {
                    "label": "保证合同编号报告状态",
                    "type": "string",
                },
                "responsibility_amount_status": {
                    "label": "还款责任金额报告状态",
                    "type": "string",
                },
                "due_date_status": {
                    "label": "到期日报告状态",
                    "type": "string",
                },
                "continuation_complete": {
                    "label": "续行是否完整",
                    "type": "boolean",
                },
                "attachment_account_id": {"label": "附件账户记录ID", "type": "string"},
                "source_sequence": {"label": "源序号", "type": "integer"},
                "attachment_record_type": {"label": "附件记录类型", "type": "string"},
                "account_status": {"label": "结清状态", "type": "string"},
                "attachment_detail_id": {"label": "附件信贷明细ID", "type": "string"},
                "amount": {
                    "label": "金额",
                    "type": "money",
                    "definition": "源报告中的业务金额；具体口径由所在数据集及 amount_kind（如有）确定。",
                },
                "advance_flag": {"label": "垫款标志", "type": "string"},
                "special_transaction_id": {"label": "特定交易ID", "type": "string"},
                "transaction_type": {"label": "交易类型", "type": "string"},
                "transaction_date": {"label": "交易日期", "type": "date"},
                "transaction_amount": {"label": "交易金额", "type": "money"},
                "due_date_change_months": {
                    "label": "到期日期变更月数",
                    "type": "integer",
                },
                "transaction_detail": {"label": "交易明细信息", "type": "string"},
            }
        )
        fields.update(
            {
                "cover_subject_name": {"label": "封面企业名称", "type": "string"},
                "identity_subject_name": {"label": "身份标识企业名称", "type": "string"},
                "cover_zhongzheng_code": {
                    "label": "封面中征码",
                    "type": "long_id",
                    "sensitive": True,
                },
                "cover_unified_social_credit_code": {
                    "label": "封面统一社会信用代码",
                    "type": "long_id",
                    "sensitive": True,
                },
                "subject_name_assertion_status": {
                    "label": "封面与身份标识名称一致性",
                    "type": "string",
                },
                "creditor_institution": {"label": "债权机构", "type": "string"},
                "receive_date": {"label": "接收日期", "type": "date"},
                "last_repayment_amount": {"label": "最近一次还款总额", "type": "money"},
                "last_repayment_amount_status": {
                    "label": "最近一次还款总额报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "original_creditor_name": {"label": "初始债权人名称", "type": "string"},
                "original_claim_type": {"label": "原债权种类", "type": "string"},
                "original_debt_type": {"label": "源原债权种类", "type": "string"},
                "original_debt_type_status": {
                    "label": "源原债权种类报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "disposal_completion_date": {"label": "处置完成日期", "type": "date"},
                "settlement_date": {"label": "结清日期", "type": "date"},
                "guarantee_contract_identifier": {
                    "label": "保证合同编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "responsibility_currency": {"label": "还款责任金额币种", "type": "string"},
                "responsibility_amount_unit": {"label": "还款责任金额单位", "type": "string"},
                "obligation_currency": {"label": "借款/授信金额币种", "type": "string"},
                "obligation_amount_unit": {"label": "借款/授信金额单位", "type": "string"},
                "open_or_receive_date": {"label": "开立/接收日期", "type": "date"},
                "account_annotation_id": {"label": "账户标注ID", "type": "string"},
                "annotation_type": {
                    "label": "标注类型",
                    "type": "string",
                    "enum_ref": "account_annotation_type",
                },
                "annotation_type_label": {"label": "标注类型名称", "type": "string"},
                "issuer": {"label": "说明/声明主体", "type": "string"},
                "annotation_date": {"label": "说明/声明日期", "type": "date"},
                "annotation_date_status": {
                    "label": "说明/声明日期报告状态",
                    "type": "string",
                    "enum_ref": "source_state",
                },
                "annotation_content": {"label": "账户标注内容", "type": "text"},
                "responsibility_group_detail_id": {
                    "label": "分组还款责任明细ID",
                    "type": "string",
                },
                "transaction_group": {"label": "交易分组", "type": "string"},
                "source_group_account_count": {"label": "源分组业务笔数", "type": "integer"},
                "credit_detail_group_id": {"label": "信贷明细分组ID", "type": "string"},
                "group_phase": {"label": "信贷明细阶段", "type": "string"},
                "group_kind": {"label": "信贷明细分组类型", "type": "string"},
                "represented_dataset": {"label": "承载业务记录的数据集", "type": "string"},
                "reported_record_count": {"label": "源报告声明笔数", "type": "integer"},
                "reported_record_count_status": {
                    "label": "源报告声明笔数状态",
                    "type": "string",
                },
                "reported_record_count_conflicts": {
                    "label": "冲突的源报告声明笔数",
                    "type": "array",
                },
            }
        )
        for status_field, label in {
            "receive_date_status": "接收日期报告状态",
            "original_creditor_name_status": "初始债权人名称报告状态",
            "original_claim_type_status": "原债权种类报告状态",
            "loan_amount_status": "借款金额报告状态",
            "guarantee_amount_status": "担保金额报告状态",
            "balance_status": "余额报告状态",
            "overdue_total_status": "逾期总额报告状态",
            "overdue_principal_status": "逾期本金报告状态",
        }.items():
            fields.setdefault(
                status_field,
                {
                    "label": label,
                    "type": "string",
                    "enum_ref": "source_state",
                },
            )
        dictionary.setdefault("datasets", {})["enterprise_credit_detail_groups"] = {
            "definition": (
                "一行对应信贷记录明细中的一个‘共N笔’业务分组标题；"
                "物理分页和重复表示不产生重复业务分组。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "credit_detail_group_id",
                    "sequence",
                    "group_phase",
                    "business_category",
                    "group_kind",
                    "represented_dataset",
                    "reported_record_count",
                    "reported_record_count_status",
                    "reported_record_count_conflicts",
                    "source_page",
                    "source_page_end",
                    "source_table_id",
                    "source_table_id_end",
                )
            },
        }
        credit_accounts = dictionary.setdefault("datasets", {}).setdefault(
            "enterprise_credit_accounts", {}
        )
        credit_accounts["definition"] = "一行对应企业报告信贷记录明细中的一个当前或已结清账户卡片。"
        credit_accounts["aggregation"] = "金额必须按币种和金额单位分组；不得与授信额度或概要合计相加。"
        account_columns = credit_accounts.setdefault("columns", {})
        for key in (
            "sequence",
            "account_id",
            "account_type",
            "account_state",
            "activation_state",
            "business_category",
            "account_identifier",
            "institution",
            "business_type",
            "status",
            "open_date",
            "due_date",
            "maturity_date",
            "close_date",
            "receive_date",
            "snapshot_date",
            "currency",
            "amount_unit",
            "loan_amount",
            "credit_limit",
            "discount_amount",
            "guarantee_amount",
            "balance",
            "risk_exposure_amount",
            "deposit_ratio",
            "credit_limit_status",
            "loan_amount_status",
            "balance_status",
        ):
            account_columns[key] = fields[key]
        for key in (
            "business_category",
            "issuance_form",
            "guarantee_type",
            "five_tier_class",
            "overdue_total",
            "current_overdue",
            "overdue_principal",
            "overdue_months",
            "current_overdue_status",
            "last_repayment_date",
            "repayment_method",
            "remaining_periods",
            "special_transaction",
            "credit_agreement_identifier",
            "history_status",
            "original_creditor_name",
            "original_claim_type",
            "creditor_institution",
            "last_repayment_amount",
            "last_repayment_amount_status",
            "original_debt_type",
            "original_debt_type_status",
            "payoff_state",
        ):
            account_columns[key] = fields[key]
        for key in (
            "receive_date_status",
            "original_creditor_name_status",
            "original_claim_type_status",
        ):
            account_columns[key] = fields[key]
        credit_lines = dictionary["datasets"].setdefault("enterprise_credit_facilities", {})
        credit_lines["definition"] = "一行对应企业报告中的一份授信信息明细卡片；概要额度不作为记录导出。"
        credit_lines["non_additive_with"] = [
            "enterprise_credit_accounts",
            "credit_summary.facility_summary",
        ]
        line_columns = credit_lines.setdefault("columns", {})
        line_columns.update(
            {
                "institution": fields["institution"],
                "credit_line_id": fields["credit_line_id"],
                "facility_type": fields["facility_type"],
                "account_state": fields["account_state"],
                "credit_agreement_identifier": fields["credit_agreement_identifier"],
                "facility_product": {"label": "授信额度类型", "type": "string"},
                "revolving_flag": {"label": "额度循环标志", "type": "boolean"},
                "effective_date": {"label": "生效日期", "type": "date"},
                "due_date": {
                    "label": "原始兼容到期日",
                    "type": "date",
                    "deprecated": True,
                    "canonical_field": "maturity_date",
                },
                "snapshot_date": {"label": "信息截至日期", "type": "date"},
                "maturity_date": fields["maturity_date"],
                "total_limit": fields["total_limit"],
                "total_limit_status": fields["total_limit_status"],
                "used_limit": fields["used_limit"],
                "used_limit_status": fields["used_limit_status"],
                "available_limit": fields["available_limit"],
                "available_limit_status": fields["available_limit_status"],
                "payoff_state": fields["payoff_state"],
                "facility_limit": {"label": "授信限额", "type": "money"},
                "limit_identifier": {
                    "label": "授信限额编号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "amount_unit": fields["amount_unit"],
                "currency": fields["currency"],
            }
        )
        for field_name, label in {
            "economic_type": "经济类型",
            "organization_type": "组织机构类型",
            "enterprise_scale": "企业规模",
            "industry": "所属行业",
            "establishment_year": "成立年份",
            "registration_certificate_valid_through": "登记证书有效截止日期",
            "registered_address": "登记地址",
            "operating_address": "办公/经营地址",
            "operating_status": "存续状态",
        }.items():
            fields[f"{field_name}_status"] = {
                "label": f"{label}报告状态",
                "type": "string",
                "enum_ref": "source_state",
                "display": "hidden",
            }
            fields[f"{field_name}_source_institution"] = {
                "label": f"{label}信息来源机构",
                "type": "string",
            }
            fields[f"{field_name}_source_institution_status"] = {
                "label": f"{label}信息来源机构状态",
                "type": "string",
                "enum_ref": "source_state",
                "display": "hidden",
            }
        account_columns["discount_amount_status"] = fields["discount_amount_status"]
        repayment_liabilities = dictionary["datasets"].setdefault(
            "enterprise_repayment_responsibility_accounts",
            {},
        )
        repayment_liabilities["definition"] = "一行对应企业报告相关还款责任信息明细中的一个账户；跨页续表合并为同一行。"
        repayment_liability_columns = repayment_liabilities.setdefault("columns", {})
        for key in (
            "sequence",
            "account_identifier",
            "responsibility_type",
            "contract_number",
            "guarantee_contract_identifier",
            "currency",
            "amount_unit",
            "responsibility_currency",
            "responsibility_amount_unit",
            "obligation_currency",
            "obligation_amount_unit",
            "responsibility_amount",
            "institution",
            "business_type",
            "open_date",
            "open_or_receive_date",
            "due_date",
            "loan_or_credit_amount",
            "balance",
            "five_tier_class",
            "overdue_total",
            "overdue_principal",
            "overdue_months_or_repayment_status",
            "remaining_periods",
            "snapshot_date",
            "contract_number_status",
            "responsibility_amount_status",
            "due_date_status",
            "continuation_complete",
        ):
            if key in fields:
                repayment_liability_columns[key] = fields[key]
        dictionary["datasets"]["enterprise_account_annotations"] = {
            "definition": (
                "一行对应正文中绑定到一个账户的数据提供机构说明、信息主体声明、"
                "征信中心说明或异议处理状态。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "account_annotation_id",
                    "sequence",
                    "account_id",
                    "account_identifier",
                    "annotation_type",
                    "annotation_type_label",
                    "issuer",
                    "annotation_date",
                    "annotation_date_status",
                    "annotation_content",
                    "dispute_status",
                    "source_page",
                    "source_page_end",
                )
            },
        }
        dictionary["datasets"]["enterprise_repayment_responsibility_group_details"] = {
            "definition": (
                "一行对应相关还款责任中没有独立账户编号的一条源分组明细；"
                "跨页拆开的文字单元在同一行重建。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "responsibility_group_detail_id",
                    "sequence",
                    "transaction_group",
                    "responsibility_type",
                    "contract_number",
                    "contract_number_status",
                    "responsibility_amount",
                    "responsibility_amount_status",
                    "institution",
                    "business_type",
                    "five_tier_class",
                    "account_count",
                    "loan_amount",
                    "loan_amount_status",
                    "guarantee_amount",
                    "guarantee_amount_status",
                    "amount_kind",
                    "balance",
                    "balance_status",
                    "overdue_total",
                    "overdue_total_status",
                    "overdue_principal",
                    "overdue_principal_status",
                    "source_group_account_count",
                    "currency",
                    "amount_unit",
                    "source_page",
                    "source_page_end",
                    "source_table_id",
                    "source_table_id_end",
                )
            },
        }
        dictionary["datasets"]["enterprise_report_metadata"] = {
            "definition": "一行对应企业信用报告封面的报告版本、编号、查询机构和报告时间。",
            "columns": {
                "sequence": fields["sequence"],
                "report_edition": fields["report_edition"],
                "report_number": fields["report_number"],
                "query_institution": fields["query_institution"],
                "report_time": fields["report_time"],
                "source_page": fields["source_page"],
            },
        }
        dictionary["datasets"]["enterprise_exchange_rates"] = {
            "definition": "一行对应报告说明页列示的一组汇率及有效期。",
            "columns": {
                "sequence": fields["sequence"],
                "exchange_rate_usd_cny": fields["exchange_rate_usd_cny"],
                "exchange_rate_effective_period": fields["exchange_rate_effective_period"],
                "source_page": fields["source_page"],
            },
        }
        dictionary["datasets"]["enterprise_report_identity"] = {
            "definition": "一行对应企业报告封面及身份标识表中的同一报告主体。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_identity_id",
                    "sequence",
                    "enterprise_name",
                    "subject_name",
                    "cover_subject_name",
                    "identity_subject_name",
                    "subject_name_assertion_status",
                    "zhongzheng_code",
                    "cover_zhongzheng_code",
                    "unified_social_credit_code",
                    "cover_unified_social_credit_code",
                    "organization_code",
                    "institution_credit_code",
                    "national_tax_id",
                    "local_tax_id",
                    "business_registration_number",
                    "source_page",
                    "source_page_end",
                )
            },
        }
        dictionary["datasets"]["enterprise_section_presence"] = {
            "definition": (
                "规范PBOC企业报告的一个业务章节一行；状态区分章节有记录、"
                "章节明确无记录和源报告不含该章节。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "section_presence_id",
                    "sequence",
                    "section_key",
                    "section_title",
                    "presence_status",
                    "source_state",
                    "heading_detected",
                    "record_count",
                    "source_page",
                    "source_page_end",
                )
            },
        }
        dictionary["datasets"]["enterprise_dispute_overview"] = {
            "definition": "一行对应报告身份标识后的异议提示概要。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_dispute_overview_id",
                    "sequence",
                    "in_progress_dispute_count",
                    "dispute_status",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_credit_overview"] = {
            "definition": "一行对应信息概要中的企业整体信用概览。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_credit_overview_id",
                    "sequence",
                    "first_credit_year",
                    "first_credit_year_status",
                    "credit_institution_count",
                    "active_credit_institution_count",
                    "first_repayment_responsibility_year",
                    "first_repayment_responsibility_year_status",
                    "credit_balance",
                    "credit_attention_balance",
                    "credit_adverse_balance",
                    "guarantee_balance",
                    "guarantee_attention_balance",
                    "guarantee_adverse_balance",
                    "recovered_debt_balance",
                    "currency",
                    "amount_unit",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_public_record_counts"] = {
            "definition": "信息概要中一个公共或非信贷记录类型的报告条数一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_public_record_count_id",
                    "sequence",
                    "record_type",
                    "record_count",
                    "count_scope",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_recovery_summary"] = {
            "definition": "一行对应信息概要中的一种被追偿业务及其结清状态。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_recovery_summary_id",
                    "sequence",
                    "settlement_status",
                    "recovery_type",
                    "account_count",
                    "balance",
                    "amount",
                    "latest_disposal_date",
                    "latest_repayment_date",
                    "disposal_completion_date",
                    "settlement_date",
                    "currency",
                    "amount_unit",
                    "source_page",
                    "source_table_id",
                )
            },
        }
        dictionary["datasets"]["enterprise_overdue_summary"] = {
            "definition": "一行对应信息概要列示的逾期本金、利息及其他和总额。",
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_overdue_summary_id",
                    "sequence",
                    "overdue_principal",
                    "overdue_interest_and_other",
                    "overdue_total",
                    "currency",
                    "amount_unit",
                    "source_page",
                    "source_table_id",
                )
            },
        }
        dictionary["datasets"]["enterprise_facility_summary"] = {
            "definition": "一行对应信息概要中的一个授信额度类别，金额单位为人民币万元。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "credit_line_id",
                    "facility_type",
                    "total_limit",
                    "used_limit",
                    "available_limit",
                    "utilization_rate",
                    "currency",
                    "amount_unit",
                )
            },
        }
        dictionary["datasets"]["enterprise_profile"] = {
            "definition": (
                "一行对应一个企业基本信息快照；每个业务字段使用同名的"
                "*_source_institution列保留报告中的信息来源机构。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "enterprise_profile_id",
                    "sequence",
                    "economic_type",
                    "organization_type",
                    "enterprise_scale",
                    "industry",
                    "establishment_year",
                    "registration_certificate_valid_through",
                    "registered_address",
                    "operating_address",
                    "operating_status",
                    "economic_type_status",
                    "organization_type_status",
                    "enterprise_scale_status",
                    "industry_status",
                    "establishment_year_status",
                    "registration_certificate_valid_through_status",
                    "registered_address_status",
                    "operating_address_status",
                    "operating_status_status",
                    "economic_type_source_institution",
                    "organization_type_source_institution",
                    "enterprise_scale_source_institution",
                    "industry_source_institution",
                    "establishment_year_source_institution",
                    "registration_certificate_valid_through_source_institution",
                    "registered_address_source_institution",
                    "operating_address_source_institution",
                    "operating_status_source_institution",
                    "economic_type_source_institution_status",
                    "organization_type_source_institution_status",
                    "enterprise_scale_source_institution_status",
                    "industry_source_institution_status",
                    "establishment_year_source_institution_status",
                    "registration_certificate_valid_through_source_institution_status",
                    "registered_address_source_institution_status",
                    "operating_address_source_institution_status",
                    "operating_status_source_institution_status",
                )
            },
        }
        dictionary["datasets"]["enterprise_capital_summary"] = {
            "definition": (
                "一行对应注册资本及主要出资人源表；注册资本金额与"
                "主要出资人记录可用状态分别表达，不以“无记录”覆盖资本金额。"
            ),
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "registered_capital_amount",
                    "currency",
                    "amount_unit",
                    "contributor_count",
                    "contributor_status",
                    "source_institution",
                    "update_date",
                    "source_page",
                    "contributor_source_page",
                )
            },
        }
        stakeholder_columns = {
            key: fields[key]
            for key in (
                "sequence",
                "role",
                "name",
                "identity_type",
                "identity_number",
                "ownership_percentage",
                "page",
                "source_institution",
                "update_date",
            )
            if key in fields
        }
        dictionary["datasets"]["enterprise_contributors"] = {
            "definition": "注册资本及主要出资人表中的一个出资方一行。",
            "columns": deepcopy(stakeholder_columns),
        }
        dictionary["datasets"]["enterprise_key_personnel"] = {
            "definition": "主要组成人员表中的一个职位/人员一行。",
            "columns": deepcopy(stakeholder_columns),
        }
        dictionary["datasets"]["enterprise_relationships"] = {
            "definition": "企业关联关系表中的一个关系一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "relationship_type",
                    "name",
                    "identity_type",
                    "identity_number",
                    "source_institution",
                    "update_date",
                )
                if key in fields
            },
        }
        dictionary["datasets"].pop("public_records", None)
        from docmirror.plugins.credit_report.enterprise_native.extraction import (
            enterprise_public_record_dataset_specs,
        )

        for spec in enterprise_public_record_dataset_specs().values():
            typed_columns = {
                "sequence": fields["sequence"],
                "public_record_id": fields["public_record_id"],
                **spec["columns"],
                "source_page": fields["source_page"],
                "source_table_id": fields["source_table_id"],
            }
            dictionary["datasets"][spec["dataset_id"]] = {
                "definition": spec["definition"],
                "columns": typed_columns,
            }
        dictionary["datasets"]["enterprise_closed_credit_summary"] = {
            "definition": "已结清信贷信息概要中的一个业务类别或合计行一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "closed_summary_id",
                    "transaction_group",
                    "business_category",
                    "normal_account_count",
                    "attention_account_count",
                    "adverse_account_count",
                    "total_account_count",
                    "is_total",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_current_credit_summary"] = {
            "definition": "当前信贷信息概要中的一个业务类别或合计行一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "current_summary_id",
                    "transaction_group",
                    "business_category",
                    "normal_account_count",
                    "normal_balance",
                    "attention_account_count",
                    "attention_balance",
                    "adverse_account_count",
                    "adverse_balance",
                    "total_account_count",
                    "total_balance",
                    "currency",
                    "amount_unit",
                    "is_total",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_displayed_credit_summary"] = {
            "definition": (
                "一行对应信贷记录明细中按机构、业务种类和五级分类展示的一条源汇总行；"
                "源报告金额与逐笔明细分别保留，不强制调平。"
            ),
            "aggregation": "仅用于复现源报告明细分组汇总，不得与账户明细、附件明细或信息概要相加。",
            "non_additive_with": [
                "enterprise_credit_accounts",
                "enterprise_attachment_credit_details",
                "enterprise_current_credit_summary",
                "enterprise_closed_credit_summary",
            ],
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "displayed_summary_id",
                    "settlement_status",
                    "transaction_group",
                    "business_category",
                    "institution",
                    "business_type",
                    "five_tier_class",
                    "source_group_account_count",
                    "source_account_count",
                    "source_reported_amount",
                    "amount_kind",
                    "overdue_total",
                    "overdue_principal",
                    "advance_flag",
                    "summary_scope",
                    "currency",
                    "amount_unit",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_repayment_responsibility_summary"] = {
            "definition": "相关还款责任信息概要中的一种责任类型或合计行一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "responsibility_summary_id",
                    "responsibility_type",
                    "transaction_group",
                    "recovered_responsibility_amount",
                    "recovered_account_count",
                    "recovered_balance",
                    "other_credit_responsibility_amount",
                    "other_credit_account_count",
                    "other_credit_balance",
                    "other_credit_attention_balance",
                    "other_credit_adverse_balance",
                    "guarantee_responsibility_amount",
                    "guarantee_account_count",
                    "guarantee_balance",
                    "guarantee_attention_balance",
                    "guarantee_adverse_balance",
                    "currency",
                    "amount_unit",
                    "is_total",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_attachment_accounts"] = {
            "definition": "信用记录补充信息中的一个账户或已结清业务标题一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "attachment_account_id",
                    "source_sequence",
                    "attachment_record_type",
                    "account_identifier",
                    "institution",
                    "business_type",
                    "business_category",
                    "account_status",
                    "five_tier_class",
                    "source_page",
                    "source_page_end",
                )
            },
        }
        dictionary["datasets"]["enterprise_credit_supplement"] = {
            "definition": "信用记录补充信息中一个账户在一个信息报告日期的完整月度记录一行；由相邻物理行规范合并。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "supplement_id",
                    "attachment_account_id",
                    "account_id",
                    "account_identifier",
                    "institution",
                    "business_type",
                    "business_category",
                    "account_status",
                    "report_date",
                    "balance",
                    "balance_change_date",
                    "five_tier_class",
                    "classification_date",
                    "overdue_total",
                    "overdue_principal",
                    "overdue_months",
                    "scheduled_repayment_date",
                    "scheduled_repayment_amount",
                    "actual_repayment_date",
                    "actual_repayment_amount",
                    "repayment_method",
                    "remaining_periods",
                    "currency",
                    "amount_unit",
                    "source_page",
                    "source_table_id",
                )
            },
        }
        dictionary["datasets"]["enterprise_attachment_credit_details"] = {
            "definition": "信用记录补充信息中一条信贷明细或担保业务账户明细一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "attachment_detail_id",
                    "attachment_account_id",
                    "account_identifier",
                    "institution",
                    "business_type",
                    "business_category",
                    "account_status",
                    "open_date",
                    "due_date",
                    "currency",
                    "amount",
                    "amount_kind",
                    "credit_limit",
                    "credit_limit_status",
                    "loan_amount",
                    "loan_amount_status",
                    "discount_amount",
                    "discount_amount_status",
                    "instrument_amount",
                    "instrument_amount_status",
                    "guarantee_amount",
                    "guarantee_amount_status",
                    "amount_unit",
                    "close_date",
                    "guarantee_type",
                    "counter_guarantee_type",
                    "deposit_ratio",
                    "balance",
                    "risk_exposure_amount",
                    "five_tier_class",
                    "five_tier_class_source",
                    "credit_agreement_identifier",
                    "credit_agreement_status",
                    "snapshot_date",
                    "last_repayment_date",
                    "repayment_method",
                    "advance_flag",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_special_transactions"] = {
            "definition": "信用记录补充信息中特定交易提示的一项交易一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "special_transaction_id",
                    "attachment_account_id",
                    "account_identifier",
                    "institution",
                    "business_type",
                    "business_category",
                    "transaction_type",
                    "transaction_date",
                    "transaction_amount",
                    "due_date_change_months",
                    "transaction_detail",
                    "currency",
                    "amount_unit",
                    "source_page",
                )
            },
        }
        dictionary["datasets"]["enterprise_interest_arrears"] = {
            "definition": "未结清信贷欠息表中的一个机构欠息记录一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "interest_arrears_id",
                    "sequence",
                    "institution",
                    "arrears_type",
                    "currency",
                    "amount_unit",
                    "arrears_balance",
                    "balance_change_date",
                    "snapshot_date",
                    "source_page",
                    "source_table_id",
                )
            },
        }
        history_columns = {
            key: fields[key]
            for key in (
                "sequence",
                "statistics_month",
                "payment_status",
                "amount_due",
                "amount_paid",
                "cumulative_arrears",
                "currency",
                "amount_unit",
                "source_page",
                "source_table_id",
            )
        }
        dictionary["datasets"]["enterprise_utility_payment_history"] = {
            "definition": "附件中一个公用事业统计月份的缴费记录一行。",
            "columns": deepcopy(history_columns),
        }
        dictionary["datasets"]["enterprise_housing_fund_history"] = {
            "definition": "附件中一个住房公积金统计月份的缴费记录一行。",
            "columns": deepcopy(history_columns),
        }
        enums = dictionary.setdefault("enums", {})
        enums.setdefault("account_type", {})["enterprise_credit"] = "企业信贷账户"
        enums["facility_type"] = {
            "non_revolving": "非循环信用额度",
            "revolving": "循环信用额度",
        }
        enums["record_type"] = {
            "non_credit_accounts": "非信贷交易账户",
            "utility_payment": "公用事业缴费",
            "tax_arrears": "欠税记录",
            "civil_judgment": "民事判决",
            "enforcement": "强制执行",
            "administrative_penalty": "行政处罚",
            "housing_fund_payment": "住房公积金缴费记录",
            "social_security_payment": "社会保险缴费记录",
            "license": "许可记录",
            "certification": "认证记录",
            "qualification": "资质记录",
            "award": "奖励记录",
            "export_quality": "出入境检验检疫绿色通道信息",
            "inspection_exemption": "进出口商品免检信息",
            "regulatory_supervision": "进出口免检分类监管信息",
            "patent": "专利记录",
            "financing_restriction": "融资规模控制信息",
            "data_provider_statement": "数据提供机构说明",
            "credit_bureau_statement": "征信中心说明",
            "subject_statement": "信息主体声明",
            "dispute_annotation": "异议标注",
        }
        enums["account_annotation_type"] = {
            "data_provider_statement": "数据提供机构说明",
            "subject_statement": "信息主体声明",
            "credit_bureau_statement": "征信中心说明",
            "dispute_processing": "异议处理中",
        }
        enums["dispute_status"] = {
            "in_progress": "处理中",
            "resolved": "已处理",
            "not_applicable": "不适用",
        }
        enums["amount_unit"] = {
            "CNY_1": "元（人民币）",
            "USD_1": "美元",
            "EUR_1": "欧元",
            "HKD_1": "港币",
            "CNY_10K": "万元（人民币）",
            "USD_10K": "万美元",
            "EUR_10K": "万欧元",
            "HKD_10K": "万港币",
            "1": "元（源币种）",
            "10K": "万（源币种）",
        }
        enums["responsibility_amount_unit"] = dict(enums["amount_unit"])
        enums["obligation_amount_unit"] = dict(enums["amount_unit"])
        enums["currency"] = {
            "CNY": "人民币",
            "USD": "美元",
            "EUR": "欧元",
            "HKD": "港币",
        }
        enums["account_state"] = {
            "open": "未关闭",
            "closed": "已关闭",
            "unknown": "未知",
        }
        enums["activation_state"] = {
            "active": "已激活",
            "inactive": "未激活",
            "not_applicable": "不适用",
            "not_reported": "未报告",
        }
        enums["current_overdue"] = {"true": "是", "false": "否"}
        enums["payoff_state"] = {
            "outstanding": "未结清",
            "settled": "已结清",
            "not_applicable": "不适用",
            "unknown": "未知",
        }
        enums["contract_number_status"] = {
            "reported": "已报告",
            "not_reported": "未报告",
        }
        enums["responsibility_amount_status"] = {
            "reported": "已报告",
            "not_reported": "未报告",
        }
        enums["due_date_status"] = {
            "reported": "已报告",
            "not_reported": "未报告",
        }
        enums["credit_agreement_status"] = {
            "reported": "已报告",
            "not_reported": "未报告",
            "not_applicable": "不适用",
        }
        enums["five_tier_class_source"] = {
            "detail_table": "信贷明细表",
            "parent_attachment_heading": "附件业务标题",
        }
        enums["continuation_complete"] = {"true": "是", "false": "否"}
        enums["continuation_family"] = {
            "current_credit_summary": "当前信贷信息概要",
            "closed_credit_summary": "已结清信贷信息概要",
            "repayment_responsibility_summary": "相关还款责任概要",
            "repayment_liability": "相关还款责任明细",
            "attachment_account": "附件账户/业务",
            "attachment_credit_detail": "附件信贷明细",
        }
        enums["reconciliation_status"] = {
            "complete": "完整",
            "unresolved": "存在未解析记录",
        }
        enums["relationship_type"] = {
            "actual_controller": "实际控制人",
            "related_enterprise": "关联企业",
        }
        enums["contributor_status"] = {
            "no_records": "无记录",
            "reported": "已报告",
        }
        enums["report_edition"] = {"independent_query": "自主查询版"}
        enums["current_overdue_status"] = {
            "overdue": "存在逾期",
            "not_overdue": "未逾期",
            "not_reported": "未报告",
            "partially_reported": "部分报告",
        }
        enums["first_repayment_responsibility_year_status"] = {"not_reported": "未报告"}
        enums["revolving_flag"] = {"true": "是", "false": "否"}
        enums["account_population_comparable"] = {"true": "是", "false": "否"}
        enums["source_display_limited"] = {"true": "是", "false": "否"}
        enums["account_dataset_scope"] = {
            "main_report_account_cards": "报告正文账户卡片",
        }
        enums["attachment_record_type"] = {
            "account": "账户",
            "business": "业务",
        }
        enums["account_status"] = {
            "active": "未结清",
            "settled": "已结清",
        }
        enums["settlement_status"] = {
            "active": "未结清",
            "settled": "已结清",
        }
        enums["group_phase"] = {
            "active": "未结清信贷",
            "recovered": "被追偿",
            "settled": "已结清信贷",
            "repayment_responsibility": "相关还款责任",
        }
        enums["recovery_type"] = {
            "asset_management_disposed_debt": "资产管理公司处置债务",
            "advance": "垫款",
        }
        enums["transaction_group"] = {
            "discount": "贴现",
            "guarantee": "担保交易",
        }
        enums["amount_kind"] = {
            "amount": "未分类金额",
            "balance": "余额",
            "credit_limit": "信用额度",
            "loan_amount": "借款金额",
            "discount_amount": "贴现金额",
            "instrument_amount": "票据/信用证金额",
            "guarantee_amount": "担保金额",
            "not_applicable": "不适用",
        }
        enums["summary_scope"] = {
            "displayed_detail_section": "信贷记录明细展示分组",
        }
        enums["status"] = {
            "active": "未结清",
            "settled": "已结清",
            "inactive": "非活动",
        }
        datasets = dictionary.setdefault("datasets", {})
        datasets["enterprise_profile"]["definition"] = (
            "一行对应一个企业基本信息快照；字段级状态明确区分源缺失值，"
            "每个业务字段的*_source_institution列保留报告中的信息来源机构。"
        )
        maturity_column = {
            "label": "到期日期",
            "type": "date",
            "definition": "合同或授信协议的到期日；不是许可或证书有效期。",
        }
        for name in (
            "enterprise_credit_accounts",
            "enterprise_credit_facilities",
            "enterprise_repayment_responsibility_accounts",
            "enterprise_attachment_credit_details",
        ):
            columns = datasets[name]["columns"]
            columns.pop("due_date", None)
            columns["maturity_date"] = dict(maturity_column)
        liability_columns = datasets["enterprise_repayment_responsibility_accounts"]["columns"]
        liability_columns.pop("due_date_status", None)
        liability_columns["maturity_date_status"] = {
            "label": "到期日报告状态",
            "type": "string",
        }
        datasets["enterprise_report_identity"]["columns"].pop("enterprise_name", None)
        for name in (
            "account_population_comparable",
            "audit_id",
            "canonical_table_account_count",
            "continuation_family",
            "credit_line_count",
            "due_date",
            "due_date_status",
            "enterprise_name",
            "expected_record_count",
            "extracted_account_count",
            "extracted_record_count",
            "projected_account_count",
            "reconciliation_status",
            "source_display_limited",
            "unexpected_record_count",
            "unresolved_record_count",
        ):
            fields.pop(name, None)
        fields.update(
            {
                "displayed_credit_account_card_count": {"label": "正文展示账户卡片数", "type": "integer"},
                "displayed_credit_facility_count": {"label": "正文展示授信协议数", "type": "integer"},
                "reported_account_count_basis": {"label": "概要账户数口径", "type": "string"},
                "source_limited_scopes": {"label": "源报告受限范围", "type": "array"},
                "source_scope_status": {"label": "源报告范围状态", "type": "string"},
                "maturity_date_status": {"label": "到期日报告状态", "type": "string"},
            }
        )
        for name in ("account_population_comparable", "due_date_status", "source_display_limited"):
            enums.pop(name, None)
        enums["maturity_date_status"] = {"reported": "已报告", "not_reported": "未报告"}
        return dictionary

    def semantic_extensions(self) -> dict[str, Any]:
        """Use an enterprise reading layout with only business-facing facts."""
        from docmirror.plugins.credit_report.enterprise_native.schema import (
            enterprise_credit_report_semantic_extensions,
        )

        semantic = enterprise_credit_report_semantic_extensions()
        semantic["dataset_document_order"] = list(
            _ENTERPRISE_DOCUMENT_DATASET_ORDER
        )
        semantic["dataset_relationships"] = {
            "enterprise_credit_facilities": {
                "relationship": "independent_enterprise_facility_records",
                "additive": False,
                "non_additive_with": ["enterprise_credit_accounts", "credit_summary.facility_summary"],
            },
            "enterprise_displayed_credit_summary": {
                "relationship": "source_reported_displayed_detail_groups",
                "additive": False,
                "non_additive_with": [
                    "enterprise_credit_accounts",
                    "enterprise_attachment_credit_details",
                    "enterprise_current_credit_summary",
                    "enterprise_closed_credit_summary",
                ],
            },
        }
        semantic["dataset_reading_columns"] = {
            "enterprise_repayment_responsibility_accounts": [
                "sequence",
                "account_identifier",
                "responsibility_type",
                "contract_number",
                "institution",
                "business_type",
                "open_date",
                "maturity_date",
                "responsibility_amount",
                "loan_or_credit_amount",
                "balance",
                "five_tier_class",
                "overdue_total",
                "overdue_principal",
                "overdue_months_or_repayment_status",
                "remaining_periods",
                "snapshot_date",
                "currency",
                "amount_unit",
            ],
        }
        semantic["enhanced_markdown"] = {
            "privacy_mode": "full",
            "show_top_document_metadata": False,
            "suppress_empty_sections": True,
            "suppress_empty_columns": True,
            "dataset_layouts": {
                "enterprise_report_metadata": {
                    "hide_title": True,
                    "columns": [
                        "report_edition",
                        "report_number",
                        "query_institution",
                        "report_time",
                    ],
                },
                "report_notes": {
                    "hidden": True,
                },
                "enterprise_exchange_rates": {
                    "columns": [
                        "exchange_rate_usd_cny",
                        "exchange_rate_effective_period",
                    ],
                },
                "enterprise_report_identity": {
                    "hidden": True,
                },
                "enterprise_credit_overview": {
                    "hidden": True,
                },
                "enterprise_public_record_counts": {
                    "columns": [
                        "record_type",
                        "record_count",
                    ],
                },
                "enterprise_dispute_overview": {
                    "columns": [
                        "in_progress_dispute_count",
                        "dispute_status",
                    ],
                },
                "enterprise_recovery_summary": {
                    "columns": [
                        "settlement_status",
                        "recovery_type",
                        "account_count",
                        "balance",
                        "amount",
                        "latest_disposal_date",
                        "latest_repayment_date",
                        "disposal_completion_date",
                        "settlement_date",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_overdue_summary": {
                    "columns": [
                        "overdue_principal",
                        "overdue_interest_and_other",
                        "overdue_total",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_capital_summary": {
                    "mode": "record_cards",
                    "hide_record_titles": True,
                    "columns": [
                        "registered_capital_amount",
                        "currency",
                        "amount_unit",
                        "contributor_count",
                        "source_institution",
                        "update_date",
                    ],
                },
                "enterprise_relationships": {
                    "mode": "record_cards",
                    "title_fields": ["relationship_type"],
                    "columns": [
                        "name",
                        "identity_type",
                        "identity_number",
                        "source_institution",
                        "update_date",
                    ],
                },
                "enterprise_facility_summary": {
                    "mode": "record_cards",
                    "title_fields": ["facility_type"],
                    "columns": [
                        "total_limit",
                        "used_limit",
                        "available_limit",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_current_credit_summary": {
                    "mode": "grouped_table",
                    "group_by": "transaction_group",
                    "group_title_prefix": "",
                    "columns": [
                        "business_category",
                        "normal_account_count",
                        "normal_balance",
                        "attention_account_count",
                        "attention_balance",
                        "adverse_account_count",
                        "adverse_balance",
                        "total_account_count",
                        "total_balance",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_closed_credit_summary": {
                    "mode": "grouped_table",
                    "group_by": "transaction_group",
                    "group_title_prefix": "",
                    "columns": [
                        "business_category",
                        "normal_account_count",
                        "attention_account_count",
                        "adverse_account_count",
                        "total_account_count",
                    ],
                },
                "enterprise_displayed_credit_summary": {
                    "mode": "grouped_table",
                    "group_by": "settlement_status",
                    "group_title_prefix": "",
                    "columns": [
                        "business_category",
                        "institution",
                        "business_type",
                        "five_tier_class",
                        "source_account_count",
                        "source_reported_amount",
                        "amount_kind",
                        "overdue_total",
                        "overdue_principal",
                        "advance_flag",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_repayment_responsibility_summary": {
                    "columns": [
                        "responsibility_type",
                        "recovered_responsibility_amount",
                        "recovered_account_count",
                        "recovered_balance",
                        "other_credit_responsibility_amount",
                        "other_credit_account_count",
                        "other_credit_balance",
                        "other_credit_attention_balance",
                        "other_credit_adverse_balance",
                        "guarantee_responsibility_amount",
                        "guarantee_account_count",
                        "guarantee_balance",
                        "guarantee_attention_balance",
                        "guarantee_adverse_balance",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_repayment_responsibility_accounts": {
                    "mode": "record_cards",
                    "title_fields": ["responsibility_type", "account_identifier"],
                    "columns": [
                        "account_identifier",
                        "contract_number",
                        "guarantee_contract_identifier",
                        "institution",
                        "business_type",
                        "open_date",
                        "open_or_receive_date",
                        "maturity_date",
                        "currency",
                        "amount_unit",
                        "responsibility_currency",
                        "responsibility_amount_unit",
                        "obligation_currency",
                        "obligation_amount_unit",
                        "responsibility_amount",
                        "loan_or_credit_amount",
                        "balance",
                        "five_tier_class",
                        "overdue_total",
                        "overdue_principal",
                        "overdue_months_or_repayment_status",
                        "remaining_periods",
                        "snapshot_date",
                    ],
                },
                "enterprise_credit_accounts": {
                    "mode": "record_cards",
                    "title_fields": ["business_category", "account_identifier"],
                    "columns": [
                        "account_identifier",
                        "institution",
                        "business_type",
                        "status",
                        "open_date",
                        "maturity_date",
                        "close_date",
                        "receive_date",
                        "snapshot_date",
                        "currency",
                        "amount_unit",
                        "loan_amount",
                        "credit_limit",
                        "discount_amount",
                        "guarantee_amount",
                        "balance",
                        "risk_exposure_amount",
                        "deposit_ratio",
                        "issuance_form",
                        "guarantee_type",
                        "five_tier_class",
                        "overdue_total",
                        "overdue_principal",
                        "overdue_months",
                        "last_repayment_date",
                        "last_repayment_amount",
                        "repayment_method",
                        "remaining_periods",
                        "special_transaction",
                        "credit_agreement_identifier",
                        "history_status",
                        "original_creditor_name",
                        "original_claim_type",
                        "creditor_institution",
                        "original_debt_type",
                    ],
                },
                "enterprise_credit_facilities": {
                    "mode": "record_cards",
                    "title_fields": ["facility_type"],
                    "columns": [
                        "facility_product",
                        "credit_agreement_identifier",
                        "institution",
                        "revolving_flag",
                        "effective_date",
                        "maturity_date",
                        "snapshot_date",
                        "total_limit",
                        "used_limit",
                        "available_limit",
                        "facility_limit",
                        "limit_identifier",
                        "currency",
                        "amount_unit",
                    ],
                },
                "enterprise_credit_supplement": {
                    "mode": "grouped_table",
                    "hide_title": True,
                    "group_by": "account_identifier",
                    "group_metadata": ["institution", "business_type", "business_category"],
                    "columns": [
                        "report_date",
                        "balance",
                        "balance_change_date",
                        "five_tier_class",
                        "classification_date",
                        "overdue_total",
                        "overdue_principal",
                        "overdue_months",
                        "scheduled_repayment_date",
                        "scheduled_repayment_amount",
                        "actual_repayment_date",
                        "actual_repayment_amount",
                        "repayment_method",
                        "remaining_periods",
                    ],
                    "column_groups": [
                        {
                            "title": "余额与风险",
                            "columns": [
                                "report_date",
                                "balance",
                                "balance_change_date",
                                "five_tier_class",
                                "classification_date",
                                "overdue_total",
                                "overdue_principal",
                                "overdue_months",
                            ],
                        },
                        {
                            "title": "还款表现",
                            "columns": [
                                "report_date",
                                "scheduled_repayment_date",
                                "scheduled_repayment_amount",
                                "actual_repayment_date",
                                "actual_repayment_amount",
                                "repayment_method",
                                "remaining_periods",
                            ],
                        },
                    ],
                },
                "enterprise_attachment_accounts": {
                    "mode": "record_cards",
                    "hide_title": True,
                    "title_fields": ["business_category", "account_identifier"],
                    "columns": [
                        "account_identifier",
                        "institution",
                        "business_type",
                        "business_category",
                        "account_status",
                        "five_tier_class",
                    ],
                },
                "enterprise_attachment_credit_details": {
                    "mode": "grouped_table",
                    "group_by": "business_category",
                    "columns": [
                        "account_identifier",
                        "institution",
                        "business_type",
                        "account_status",
                        "open_date",
                        "maturity_date",
                        "currency",
                        "amount_kind",
                        "credit_limit",
                        "loan_amount",
                        "discount_amount",
                        "instrument_amount",
                        "guarantee_amount",
                        "amount_unit",
                        "close_date",
                        "guarantee_type",
                        "counter_guarantee_type",
                        "deposit_ratio",
                        "balance",
                        "risk_exposure_amount",
                        "five_tier_class",
                        "credit_agreement_identifier",
                        "snapshot_date",
                        "last_repayment_date",
                        "repayment_method",
                        "advance_flag",
                    ],
                },
                "enterprise_special_transactions": {
                    "mode": "grouped_table",
                    "group_by": "account_identifier",
                    "group_metadata": [
                        "institution",
                        "business_type",
                        "business_category",
                    ],
                    "columns": [
                        "transaction_type",
                        "transaction_date",
                        "transaction_amount",
                        "due_date_change_months",
                        "transaction_detail",
                        "currency",
                        "amount_unit",
                    ],
                },
            },
            "section_layouts": {
                "identity": {
                    "omit_unlisted": True,
                    "groups": [
                        {
                            "hide_title": True,
                            "fields": [
                                "subject_name",
                                "zhongzheng_code",
                                "unified_social_credit_code",
                                "organization_code",
                                "business_registration_number",
                                "institution_credit_code",
                                "national_tax_id",
                                "local_tax_id",
                            ],
                        },
                    ],
                },
                "credit_summary": {
                    "omit_unlisted": True,
                    "groups": [
                        {
                            "title": "信贷概况",
                            "fields": [
                                "first_credit_year",
                                "credit_institution_count",
                                "active_credit_institution_count",
                                "first_repayment_responsibility_year",
                            ],
                        },
                        {
                            "title": "余额概览",
                            "fields": [
                                "credit_balance",
                                "credit_attention_balance",
                                "credit_adverse_balance",
                                "guarantee_balance",
                                "guarantee_attention_balance",
                                "guarantee_adverse_balance",
                                "recovered_debt_balance",
                                "amount_unit",
                            ],
                        },
                    ],
                },
            },
        }
        from docmirror.plugins.credit_report.enterprise_native.extraction import (
            enterprise_public_record_dataset_specs,
        )

        dataset_layouts = semantic["enhanced_markdown"]["dataset_layouts"]
        for spec in enterprise_public_record_dataset_specs().values():
            dataset_layouts[spec["dataset_id"]] = {
                "mode": "table",
                "columns": list(spec["columns"]),
            }
        dataset_layouts["enterprise_profile"] = {
            "mode": "record_cards",
            "hide_title": True,
            "hide_record_titles": True,
            "columns": [
                "economic_type",
                "economic_type_source_institution",
                "organization_type",
                "organization_type_source_institution",
                "enterprise_scale",
                "enterprise_scale_source_institution",
                "industry",
                "industry_source_institution",
                "establishment_year",
                "establishment_year_source_institution",
                "registration_certificate_valid_through",
                "registration_certificate_valid_through_source_institution",
                "registered_address",
                "registered_address_source_institution",
                "operating_address",
                "operating_address_source_institution",
                "operating_status",
                "operating_status_source_institution",
            ],
        }
        dataset_layouts["enterprise_section_presence"] = {
            "hidden": True,
        }
        dataset_layouts["enterprise_credit_detail_groups"] = {
            "mode": "record_cards",
            "hide_title": True,
            "title_fields": ["group_phase", "business_category"],
            "columns": ["reported_record_count"],
        }
        dataset_layouts["enterprise_account_annotations"] = {
            "mode": "record_cards",
            "title_fields": ["annotation_type_label", "account_identifier"],
            "columns": [
                "account_identifier",
                "annotation_type_label",
                "issuer",
                "annotation_date",
                "annotation_content",
                "dispute_status",
            ],
        }
        dataset_layouts["enterprise_interest_arrears"] = {
            "mode": "record_cards",
            "title_fields": ["arrears_type", "institution"],
            "columns": [
                "institution",
                "arrears_type",
                "arrears_balance",
                "balance_change_date",
                "snapshot_date",
                "currency",
                "amount_unit",
            ],
        }
        dataset_layouts["enterprise_repayment_responsibility_group_details"] = {
            "mode": "record_cards",
            "title_fields": ["transaction_group", "responsibility_type"],
            "columns": [
                "responsibility_type",
                "contract_number",
                "responsibility_amount",
                "institution",
                "business_type",
                "five_tier_class",
                "account_count",
                "loan_amount",
                "guarantee_amount",
                "amount_kind",
                "balance",
                "overdue_total",
                "overdue_principal",
                "currency",
                "amount_unit",
            ],
        }
        for history_dataset in (
            "enterprise_utility_payment_history",
            "enterprise_housing_fund_history",
        ):
            dataset_layouts[history_dataset] = {
                "mode": "table",
                "columns": [
                    "statistics_month",
                    "payment_status",
                    "amount_due",
                    "amount_paid",
                    "cumulative_arrears",
                    "currency",
                    "amount_unit",
                ],
            }
        dataset_layouts["enterprise_contributors"] = {
            "mode": "record_cards",
            "title_fields": ["role"],
            "columns": [
                "name", "identity_type", "identity_number", "ownership_percentage",
                "source_institution", "update_date",
            ],
        }
        dataset_layouts["enterprise_key_personnel"] = deepcopy(dataset_layouts["enterprise_contributors"])
        return semantic



variant = EnterpriseNativeVariant()

__all__ = ["EnterpriseNativeVariant", "variant"]

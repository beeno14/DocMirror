# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration adapter for native-text enterprise credit reports."""

from typing import Any

from docmirror.plugins.credit_report.shared.variant import CreditReportVariantAdapter


class EnterpriseNativeVariant(CreditReportVariantAdapter):
    """Keep enterprise extraction behind a dedicated variant boundary."""

    def __init__(self) -> None:
        super().__init__(
            variant_id="enterprise_native",
            report_subtype="enterprise",
            expected_content_modes=frozenset({"native_text", "mixed"}),
            include_credit_lines=True,
        )

    def use_generic_credit_accounts(self) -> bool:
        """Enterprise stacked cards replace generic row-shaped candidates."""
        return False

    def extract_native_business(
        self,
        parse_result: Any,
        full_text: str,
        *,
        content_mode: str,
    ) -> dict[str, Any]:
        """Extract only native enterprise business records."""
        if content_mode not in self.expected_content_modes:
            return {}
        from docmirror.plugins.credit_report.enterprise_native.extraction import (
            extract_enterprise_native_business,
        )

        return extract_enterprise_native_business(parse_result, full_text)

    def build_section_content(self, parse_result: Any, full_text: str) -> dict[str, Any]:
        """Return enterprise profile datasets without entering personal extractors."""
        from docmirror.plugins.credit_report.enterprise_native.extraction import (
            extract_enterprise_capital_summary,
            extract_enterprise_facility_summary,
            extract_enterprise_identity_facts,
            extract_enterprise_profile_datasets,
            extract_enterprise_report_metadata_records,
            extract_enterprise_report_notes,
            extract_enterprise_supplement_rows,
        )

        datasets = extract_enterprise_profile_datasets(parse_result)
        datasets.update(extract_enterprise_report_metadata_records(parse_result, full_text))
        datasets["enterprise_capital_summary"] = extract_enterprise_capital_summary(parse_result)
        datasets["enterprise_facility_summary"] = extract_enterprise_facility_summary(parse_result)
        datasets["enterprise_credit_supplement"] = extract_enterprise_supplement_rows(parse_result)
        return {
            "facts": extract_enterprise_identity_facts(parse_result),
            "report_notes": extract_enterprise_report_notes(parse_result),
            "datasets": datasets,
        }

    def refine_domain_facts(
        self,
        domain_facts: dict[str, Any],
        field_details: dict[str, Any],
    ) -> None:
        """Remove personal-only artifacts from the enterprise identity view."""
        for field_name in ("id_type", "id_number", "subject_id", "marital_status", "company_name"):
            domain_facts.pop(field_name, None)
            field_details.pop(field_name, None)

    def refine_entity_fields(self, entity_fields: dict[str, Any]) -> None:
        """Do not leak a personal subject identifier into enterprise entities."""
        for field_name in ("id_type", "id_number", "subject_id", "marital_status"):
            entity_fields.pop(field_name, None)

    def data_dictionary(self) -> dict[str, Any]:
        """Use enterprise labels without changing personal-report semantics."""
        dictionary = super().data_dictionary()
        fields = dictionary.setdefault("fields", {})
        for field_name in ("subject_id", "id_number", "id_type", "marital_status"):
            fields.pop(field_name, None)
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
        fields["query_institution"] = {"label": "查询机构", "type": "string"}
        fields.update(
            {
                "credit_balance": {"label": "借贷交易余额", "type": "money"},
                "guarantee_balance": {"label": "担保交易余额", "type": "money"},
                "recovered_debt_balance": {"label": "被追偿余额", "type": "money"},
                "first_credit_year": {"label": "首次有信贷交易的年份", "type": "integer"},
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
                "public_record_count": {"label": "公共记录数", "type": "integer"},
                "public_record_counts": {"label": "各类公共记录数", "type": "object"},
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
                "source_page": {"label": "源页码", "type": "integer"},
                "source_table_id": {"label": "源表标识", "type": "string"},
                "source_row_number": {"label": "源表行号", "type": "integer"},
                "column_count": {"label": "列数", "type": "integer"},
                "row_text": {"label": "源表行内容", "type": "string"},
                "credit_line_id": {"label": "授信记录ID", "type": "string"},
                "facility_type": {"label": "授信类型", "type": "string"},
                "total_limit": {"label": "总额度", "type": "money"},
                "used_limit": {"label": "已用额度", "type": "money"},
                "available_limit": {"label": "剩余可用额度", "type": "money"},
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
                "institution": {"label": "授信机构", "type": "string"},
                "business_type": {"label": "业务类型", "type": "string"},
                "business_category": {"label": "业务类别", "type": "string"},
                "report_date": {"label": "信息报告日期", "type": "date"},
                "balance": {"label": "余额", "type": "money"},
                "balance_change_date": {"label": "余额变化日期", "type": "date"},
                "five_tier_class": {"label": "五级分类", "type": "string"},
                "classification_date": {"label": "五级分类认定日期", "type": "date"},
                "overdue_total": {"label": "逾期总额", "type": "money"},
                "overdue_principal": {"label": "逾期本金", "type": "money"},
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
                "guarantee_type": {"label": "担保方式", "type": "string"},
                "status": {"label": "账户状态", "type": "string"},
                "current_overdue_amount": {"label": "当前逾期总额", "type": "money"},
                "current_overdue_periods": {"label": "当前逾期月数", "type": "integer"},
                "current_overdue_status": {
                    "label": "当前逾期报告状态",
                    "type": "string",
                },
                "last_repayment_date": {"label": "最近还款日期", "type": "date"},
                "actual_payment": {"label": "最近实还总额", "type": "money"},
                "special_transaction": {"label": "特定交易提示", "type": "string"},
                "credit_agreement_identifier": {
                    "label": "授信协议编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "history_status": {"label": "历史表现", "type": "string"},
            }
        )
        credit_accounts = dictionary.setdefault("datasets", {}).setdefault("credit_accounts", {})
        credit_accounts["definition"] = "一行对应企业报告信贷记录明细中的一个当前或已结清账户卡片。"
        credit_accounts["aggregation"] = "金额必须按币种和金额单位分组；不得与授信额度或概要合计相加。"
        account_columns = credit_accounts.setdefault("columns", {})
        account_columns["institution"] = fields["institution"]
        account_columns["status"] = fields["status"]
        for key in (
            "business_category",
            "guarantee_type",
            "five_tier_class",
            "current_overdue_amount",
            "overdue_principal",
            "current_overdue_periods",
            "current_overdue_status",
            "last_repayment_date",
            "actual_payment",
            "repayment_method",
            "remaining_periods",
            "special_transaction",
            "credit_agreement_identifier",
            "history_status",
        ):
            account_columns[key] = fields[key]
        credit_lines = dictionary["datasets"].setdefault("credit_lines", {})
        credit_lines["definition"] = "一行对应企业报告中的一份授信信息明细卡片；概要额度不作为记录导出。"
        credit_lines["non_additive_with"] = ["credit_accounts", "credit_summary.facility_summary"]
        line_columns = credit_lines.setdefault("columns", {})
        line_columns.update(
            {
                "institution": fields["institution"],
                "account_identifier": {
                    "label": "授信协议编号",
                    "type": "long_id",
                    "sensitive": True,
                },
                "facility_product": {"label": "授信额度类型", "type": "string"},
                "revolving_flag": {"label": "额度循环标志", "type": "boolean"},
                "effective_date": {"label": "生效日期", "type": "date"},
                "due_date": {"label": "到期日", "type": "date"},
                "snapshot_date": {"label": "信息报告日期", "type": "date"},
                "facility_limit": {"label": "授信限额", "type": "money"},
                "limit_identifier": {
                    "label": "授信限额编号",
                    "type": "long_id",
                    "sensitive": True,
                    "display": "masked",
                },
                "amount_unit": fields["amount_unit"],
            }
        )
        dictionary["datasets"]["enterprise_report_metadata"] = {
            "definition": "一行对应企业信用报告封面标示的报告版本。",
            "columns": {
                "sequence": fields["sequence"],
                "report_edition": fields["report_edition"],
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
        dictionary["datasets"]["enterprise_profile_fields"] = {
            "definition": "企业基本信息源表中的一个字段一行。",
            "columns": {
                key: fields[key]
                for key in ("sequence", "field", "value", "source_institution")
                if key in fields
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
                )
            },
        }
        dictionary["datasets"]["enterprise_stakeholders"] = {
            "definition": "企业出资人或主要人员表中的一个角色一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "role",
                    "name",
                    "identity_type",
                    "identity_number",
                    "ownership_percentage",
                    "source_institution",
                    "update_date",
                )
                if key in fields
            },
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
        dictionary["datasets"]["public_records"] = {
            "definition": "企业报告公共记录及声明、异议部分中的一项源记录一行。",
            "columns": {
                key: fields[key]
                for key in (
                    "sequence",
                    "public_record_id",
                    "record_type",
                    "authority",
                    "category",
                    "start_date",
                    "end_date",
                    "content",
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
                    "account_id",
                    "account_identifier",
                    "institution",
                    "business_type",
                    "business_category",
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
                    "source_page",
                    "source_table_id",
                )
            },
        }
        enums = dictionary.setdefault("enums", {})
        enums.setdefault("account_type", {})["enterprise_credit"] = "企业信贷账户"
        enums["facility_type"] = {
            "non_revolving": "非循环信用额度",
            "revolving": "循环信用额度",
        }
        enums["record_type"] = {
            "utility_payment": "公用事业缴费",
            "tax_arrears": "欠税记录",
            "civil_judgment": "民事判决",
            "enforcement": "强制执行",
            "administrative_penalty": "行政处罚",
            "social_security_payment": "社会保障/公积金缴费",
            "license": "许可记录",
            "certification": "认证记录",
            "qualification": "资质记录",
            "award": "奖励记录",
            "export_quality": "出口商品质量记录",
            "inspection_exemption": "免检记录",
            "regulatory_supervision": "监管记录",
            "patent": "专利记录",
            "financing_restriction": "融资约束名录",
            "data_provider_statement": "数据提供机构说明",
            "credit_bureau_statement": "征信中心说明",
            "subject_statement": "信息主体声明",
            "dispute_annotation": "异议标注",
        }
        enums["amount_unit"] = {"CNY_10K": "万元（人民币）"}
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
        enums["status"] = {
            "active": "未结清",
            "settled": "已结清",
            "inactive": "非活动",
        }
        return dictionary

    def semantic_extensions(self) -> dict[str, Any]:
        """Use an enterprise reading layout with only business-facing facts."""
        semantic = super().semantic_extensions()
        policy = semantic.setdefault("presentation_policy", {})
        policy["classification"] = "sensitive_enterprise_credit_data"
        policy["enhanced_markdown_display"] = "full"
        policy["mask_fields"] = [
            "report_number",
            "account_identifier",
            "contract_number",
            "limit_identifier",
            "identity_number",
        ]
        semantic["dataset_relationships"] = {
            "credit_lines": {
                "relationship": "independent_enterprise_facility_records",
                "additive": False,
                "non_additive_with": ["credit_accounts", "credit_summary.facility_summary"],
            }
        }
        semantic["audit_csv"] = {
            "reconciliations": [
                {
                    "name": "credit_account_balance",
                    "fields": [
                        "expected",
                        "actual",
                        "difference",
                        "tolerance",
                        "currency",
                        "amount_unit",
                        "matched",
                        "status",
                    ],
                }
            ]
        }
        semantic["enhanced_markdown"] = {
            "privacy_mode": "full",
            "show_top_document_metadata": False,
            "dataset_layouts": {
                "enterprise_report_metadata": {
                    "hide_title": True,
                    "columns": ["report_edition"],
                },
                "report_notes": {
                    "hide_title": True,
                    "columns": ["sequence", "content"],
                },
                "enterprise_exchange_rates": {
                    "columns": [
                        "exchange_rate_usd_cny",
                        "exchange_rate_effective_period",
                    ],
                },
                "enterprise_capital_summary": {
                    "columns": [
                        "registered_capital_amount",
                        "currency",
                        "amount_unit",
                        "contributor_count",
                        "contributor_status",
                        "source_institution",
                        "update_date",
                    ],
                },
                "enterprise_stakeholders": {
                    "mode": "record_cards",
                    "title_fields": ["role"],
                    "columns": [
                        "name",
                        "identity_type",
                        "identity_number",
                        "ownership_percentage",
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
                        "utilization_rate",
                        "currency",
                        "amount_unit",
                    ],
                },
                "credit_accounts": {
                    "mode": "record_cards",
                    "title_fields": ["sequence", "business_category"],
                    "title_separator": ". ",
                    "columns": [
                        "account_identifier",
                        "institution",
                        "business_type",
                        "status",
                        "open_date",
                        "due_date",
                        "snapshot_date",
                        "currency",
                        "amount_unit",
                        "loan_amount",
                        "credit_limit",
                        "balance",
                        "guarantee_type",
                        "five_tier_class",
                        "current_overdue_amount",
                        "overdue_principal",
                        "current_overdue_periods",
                        "current_overdue_status",
                        "last_repayment_date",
                        "actual_payment",
                        "repayment_method",
                        "remaining_periods",
                        "special_transaction",
                        "credit_agreement_identifier",
                        "history_status",
                    ],
                },
                "credit_lines": {
                    "mode": "record_cards",
                    "title_fields": ["facility_type"],
                    "columns": [
                        "facility_product",
                        "account_identifier",
                        "institution",
                        "revolving_flag",
                        "effective_date",
                        "due_date",
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
            },
            "section_layouts": {
                "identity": {
                    "omit_unlisted": False,
                    "hidden_fields": [
                        "report_subtype",
                        "content_mode",
                    ],
                },
                "credit_summary": {
                    "omit_unlisted": False,
                    "hidden_fields": [
                        "facility_summary",
                    ],
                    "hidden_groups": ["facility_summary"],
                    "groups": [
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
                        {
                            "title": "账户与授信概览",
                            "fields": [
                                "reported_account_count",
                                "reported_account_balance",
                                "reported_credit_line_count",
                                "facility_summary_record_count",
                                "public_record_count",
                                "first_credit_year",
                                "credit_institution_count",
                                "active_credit_institution_count",
                                "first_repayment_responsibility_year",
                                "first_repayment_responsibility_year_status",
                            ],
                            "nested_groups": [
                                "reported_account_counts",
                                "reported_account_balances",
                                "public_record_counts",
                            ],
                        },
                    ],
                }
            },
            "appendix": {
                "title": "附录：文档来源与审计信息",
                "audit_reconciliations": [
                    {
                        "name": "credit_account_balance",
                        "title": "源文账户余额矛盾（审计信息）",
                        "fields": [
                            {"key": "expected", "label": "源报告账户余额合计"},
                            {"key": "actual", "label": "账户明细余额计算合计"},
                            {"key": "difference", "label": "两者差额"},
                            {"key": "tolerance", "label": "审计舍入容差"},
                            {"key": "currency", "label": "币种"},
                            {"key": "amount_unit", "label": "金额单位"},
                            {
                                "key": "status",
                                "label": "审计结论",
                                "value_labels": {
                                    "exact": "原文数值一致",
                                    "within_rounding_tolerance": "存在差异，但在舍入容差内",
                                    "mismatch": "存在超出舍入容差的差异",
                                    "not_comparable": "不可比",
                                },
                            },
                        ],
                        "note": (
                            "源报告概要值与账户明细值均按原文保留；"
                            "审计计算不改写任何业务数据。"
                        ),
                    }
                ],
                "document_fields": [
                    {
                        "path": "source_file.name",
                        "key": "source_file_name",
                        "label": "源文件",
                    },
                    {
                        "path": "page_count",
                        "key": "page_count",
                        "label": "页数",
                    },
                ],
            },
        }
        return semantic

    def build_sections(self, parse_result: Any, full_text: str) -> tuple[dict[str, Any], ...]:
        """Keep enterprise sections and skip table-of-contents occurrences."""
        sections = [dict(section) for section in super().build_sections(parse_result, full_text)]
        page_texts: list[tuple[int, int, str]] = []
        for index, page in enumerate(getattr(parse_result, "pages", None) or [], start=1):
            parts = [str(getattr(block, "content", "") or "") for block in getattr(page, "texts", None) or []]
            for table in getattr(page, "tables", None) or []:
                parts.extend(str(value or "") for value in getattr(table, "headers", None) or [])
                for row in getattr(table, "rows", None) or []:
                    parts.extend(
                        str(getattr(cell, "text", "") or "")
                        for cell in getattr(row, "cells", None) or []
                    )
            page_texts.append(
                (
                    int(getattr(page, "page_number", 0) or index),
                    int(
                        getattr(page, "source_page_number", 0)
                        or getattr(page, "page_number", 0)
                        or index
                    ),
                    "\n".join(parts),
                )
            )
        source_sections: list[dict[str, Any]] = []
        cover_page = next(
            (
                (logical_page, source_page)
                for logical_page, source_page, text in page_texts
                if "自主查询版" in text
            ),
            None,
        )
        if cover_page is not None:
            source_sections.append(
                {
                    "id": "sec_enterprise_report_metadata",
                    "title": "报告信息",
                    "type": "report_metadata",
                    "page_start": cover_page[0],
                    "source_page_start": cover_page[1],
                }
            )
        notes_page = next(
            (
                (logical_page, source_page)
                for logical_page, source_page, text in page_texts
                if "报告说明" in text
            ),
            None,
        )
        if notes_page is not None:
            source_sections.append(
                {
                    "id": "sec_enterprise_report_notes",
                    "title": "说明",
                    "type": "notes",
                    "page_start": notes_page[0],
                    "source_page_start": notes_page[1],
                }
            )
        for section in sections:
            title = str(section.get("title") or "")
            if title not in {"身份标识", "信息概要", "基本信息", "信贷记录明细", "公共记录明细", "信用记录补充信息"}:
                continue
            page_start = next(
                (
                    (logical_page, source_page)
                    for logical_page, source_page, text in page_texts
                    if source_page >= 3 and title in text
                ),
                None,
            )
            if page_start is not None:
                section["page_start"] = page_start[0]
                section["source_page_start"] = page_start[1]
        return tuple([*source_sections, *sections])


variant = EnterpriseNativeVariant()

__all__ = ["EnterpriseNativeVariant", "variant"]

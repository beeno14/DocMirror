import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full/outputs/personal_brief_business_dictionary_20260808/PBOC_personal_brief_community_json_business_dictionary_中英.xlsx";
const workspace = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full";
const outputDir = `${workspace}/outputs/personal_brief_community_dictionary_audit_20260810`;
const outputPath = `${outputDir}/PBOC_personal_brief_community_json_business_and_audit_dictionary_中英.xlsx`;
const previewDir = `${workspace}/.tmp/personal_brief_dictionary_audit/previews_after`;

const colors = {
  navy: "#17365D",
  header: "#1F4E78",
  subtitle: "#D9EAF7",
  paleBlue: "#EAF5FB",
  blueBand: "#C9EAF7",
  paleGreen: "#E2F0D9",
  paleYellow: "#FFF2CC",
  paleOrange: "#FCE4D6",
  palePurple: "#E4DFEC",
  paleRed: "#FCE8E6",
  redText: "#9C0006",
  amberText: "#7F6000",
  text: "#1F2937",
  border: "#CCD6E0",
  white: "#FFFFFF",
};

const auditRows = [
  ["全局规则", "公开投影", "投影守恒", "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT", "公开投影守恒冲突", "error", "语义数据集与 Community 公开数据集、行 ID/顺序、固定 columns 及公开业务值逐项守恒。", "全部 15 个公开数据集", "dataset / record_id / columns / 固定业务字段", "语义结果与 Community JSON 存在缺失、多出、顺序差异、列合同差异或值差异。", "按 message 定位数据集、行和字段，对照 semantic JSON 与公开字段策略，确认投影遗漏或值被改变。", "message 中 dataset/record_id/fields；可选 dataset_id、section_id、page_range", "否"],
  ["全局规则", "数据集外壳", "结构一致性", "PERSONAL_BRIEF_AUDIT_DATASET_ENVELOPE_CONFLICT", "数据集封装冲突", "error", "核对 row_count、实际 rows 数量及 completeness 计数/verified 逻辑。", "全部公开数据集", "row_count；completeness.*", "数据集行数或完整性计数自相矛盾。", "重算行数和完整性封装，检查 projector 生成的数据集外壳。", "message 中 dataset/fields；dataset_id、section_id", "否"],
  ["全局规则", "数据集外壳", "主键唯一性", "PERSONAL_BRIEF_AUDIT_DUPLICATE_RECORD_ID", "记录标识重复", "error", "同一数据集内 record_id 必须唯一。", "全部公开数据集", "record_id", "同一 record_id 在一个数据集中出现多次。", "核对 ID 生成、重复投影及源记录粒度。", "message 中 dataset/record_id/fields；dataset_id、section_id", "否"],
  ["全局规则", "完整性", "完整性复核", "PERSONAL_BRIEF_AUDIT_DATASET_INCOMPLETE", "数据集完整性待复核", "warning", "检查 verified 及 omitted、unexpected、unresolved、缺必填、来源无效、边界未覆盖/重复等细项。", "对应业务数据集", "dataset.completeness；extraction_report.dataset_completeness", "数据集未被完整验证，或未验证的数据集未进入 Community JSON。", "查看 completeness 细项和重建边界，回到源 PDF 确认遗漏、未解析或来源问题。", "message 中 dataset/fields；数据集存在时有 dataset_id、section_id", "否"],
  ["全局规则", "来源追溯", "可追溯性", "PERSONAL_BRIEF_AUDIT_PROVENANCE_MISSING", "业务行来源追溯缺失", "warning", "每个公开业务行必须能在 rich semantic 中找到 source page/source_refs/evidence。", "全部公开业务行", "record_id；semantic source/provenance", "业务值已发布，但找不到任何来源证据。", "用 record_id 定位语义行并回查源 PDF；修复前不要仅凭该行自动裁决。", "message 中 dataset/record_id；dataset_id、section_id", "否"],
  ["全局规则", "提取控制", "状态一致性", "PERSONAL_BRIEF_AUDIT_EXTRACTION_REPORT_CONFLICT", "提取报告状态冲突", "error", "核对 extraction report 的 status、failures、content_conserved 和各数据集 verified。", "personal_brief_extraction_report", "status / failures / content_conserved / dataset_completeness", "提取报告声称 complete，但同时存在失败、内容不守恒或未验证数据集。", "检查提取报告生成逻辑及 failure/completeness 细项；不要直接信任 complete。", "message 中 fields=personal_brief_extraction_report", "否"],
  ["全局规则", "重建守恒", "源内容守恒", "PERSONAL_BRIEF_AUDIT_SOURCE_CONSERVATION_FAILURE", "源内容守恒失败", "warning", "检查 canonical reconstruction 是否守恒源内容，并确认失败原因已记录。", "全报告", "extraction_report.content_conserved / failures", "源内容未守恒，且没有 failure 解释。", "复核页面续接、分块边界和未覆盖源文本，确认是否丢失业务内容。", "message 中 extraction_report.content_conserved", "否"],
  ["全局规则", "枚举合同", "闭集值域", "PERSONAL_BRIEF_AUDIT_ENUM_CONTRACT_CONFLICT", "枚举合同冲突", "error", "按 Dataset + Field 核对 columns.type/enum 与每个公开枚举值。", "26 个 dataset-qualified 枚举字段", "columns.type/enum；normalized 枚举值", "列元数据与插件闭集不同，或行值不在允许值内。", "按数据集和字段核对枚举合同及源中文值；判断映射错误或合同需显式升级。", "message 中 dataset/record_id/fields；dataset_id、section_id；可选 page_range", "否"],
  ["全局规则", "金额口径", "金额单位合同", "PERSONAL_BRIEF_AUDIT_MONEY_POLICY_CONFLICT", "金额口径冲突", "error", "报告头、金额列及金额行必须统一为 CNY / CNY_1 / precision 0。", "报告头及 9 个含金额数据集", "reporting_*；money column type/unit", "全局、列级或行级金额币种、单位、精度不一致。", "先核对报告头，再核对列元数据和具体行；禁止审核端猜测缩放倍数。", "message 中 dataset/record_id/fields；dataset_id、section_id；可选 page_range", "否"],
  ["全局规则", "金额值", "金额规范", "PERSONAL_BRIEF_AUDIT_MONEY_VALUE_CONFLICT", "金额值规范冲突", "error", "16 个金额字段非空时必须为规范非负整数十进制字符串。", "全部含金额业务数据集", "PERSONAL_BRIEF_MONEY_FIELDS 中 16 个字段", "金额含负号、小数、前导零或非数字字符，不符合 CNY_1 合同。", "对照源金额文本与 CNY_1 口径，检查清洗、单位转换及字符串规范化。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["报告信息", "身份信息", "主体一致性", "PERSONAL_BRIEF_AUDIT_IDENTITY_CONFLICT", "身份信息一致性冲突", "error", "报告元数据必须恰好一行、主证件恰好一行，且姓名/证件类型/号码一致。", "personal_report_metadata；identity_documents", "subject_name / primary_id_* / is_primary / document_*", "报告头与主证件行数或主体值不一致。", "对照报告首页姓名和证件，确认重复、漏行或主证件映射问题。", "message 中 dataset/record_id/fields；dataset_id、section_id；可选 page_range", "否"],
  ["信贷记录", "信息概要", "概要一致性", "PERSONAL_BRIEF_AUDIT_SUMMARY_CONFLICT", "信息概要一致性冲突", "error", "校验 metric-category 规范格、状态和值、唯一性、计数格及具备覆盖条件的概要/详情对账。", "personal_credit_summary_records 及相关详情数据集", "metric / business_category / reporting_status / value", "概要单元格不合规、计数不可能、重复，或已验证概要数与详情数不一致。", "对照源信息概要表和对应详情；先确认 reporting_status，再核实计数粒度。", "message 中 dataset/record_id/fields；dataset_id、section_id；可选 page_range", "否"],
  ["信贷记录", "信贷账户", "账户关系", "PERSONAL_BRIEF_AUDIT_ACCOUNT_RELATION_CONFLICT", "信贷账户关系冲突", "error", "校验账户类型/业务类别、生命周期/终止事件、日期先后、结清状态、卡专属字段和逾期逻辑。", "credit_accounts", "账户类型、生命周期、日期、payoff、卡字段、逾期标志", "账户字段组合在业务上自相矛盾。", "按 record_id 对照源账户描述，逐项复核账户类型、生命周期、日期和逾期语义。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录", "信贷账户金额", "金额状态", "PERSONAL_BRIEF_AUDIT_AMOUNT_STATUS_CONFLICT", "账户金额与报告状态冲突", "error", "信用额度、已使用额度、贷款金额、余额必须与各自 *_status 成对一致。", "credit_accounts", "credit_limit/used_amount/loan_amount/balance 与对应 status", "reported 但金额缺失，或 not_reported/not_applicable 却有金额。", "对照账户源句，确认金额遗漏/误填或状态映射错误。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录 / 公共记录", "状态协议", "值-状态一致性", "PERSONAL_BRIEF_AUDIT_STATUS_VALUE_CONFLICT", "业务值与状态标志冲突", "error", "案由/行政复议结果及责任金额必须与相邻状态字段/布尔标志一致。", "相关还款责任、民事判决、强制执行、行政处罚", "value/status 字段对", "业务值存在性与 reported/not_reported 或 reported 布尔不一致。", "对照源记录确认值是否实际报告，再修正值或状态映射。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录", "账户关联", "关联键", "PERSONAL_BRIEF_AUDIT_ACCOUNT_ID_CONFLICT", "账户关联 ID 冲突", "error", "每个信贷账户必须有唯一 account_id。", "credit_accounts", "account_id", "账户缺少 account_id，或同一 account_id 对应多个账户行。", "核对账户 ID 派生键和页面续接，确保一账户一 ID。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录", "逾期关联", "外键完整性", "PERSONAL_BRIEF_AUDIT_ORPHAN_OVERDUE_RECORD", "孤立逾期记录", "error", "overdue_records.account_id 必须解析到恰好一个 credit_accounts.account_id。", "overdue_records ↔ credit_accounts", "account_id", "逾期行无法关联到唯一账户。", "先核对页面续接和账户 ID，再将逾期源块与正确账户逐一配对。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录", "逾期信息", "逾期一致性", "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT", "逾期状态一致性冲突", "error", "核对逾期行与账户身份、ever/current overdue、月份计数、90 天标志及一账户一逾期行。", "overdue_records ↔ credit_accounts", "身份字段、逾期布尔/状态/月数", "账户与逾期行在身份或逾期事实方面不一致，或月份/标志不合法。", "并排审核账户行、逾期行和源页；不要只修单侧值。", "message 中 dataset/record_id/fields；dataset_id、section_id、page_range", "否"],
  ["信贷记录", "逾期粒度", "唯一性", "PERSONAL_BRIEF_AUDIT_DUPLICATE_OVERDUE_RECORD", "逾期记录重复", "error", "同一 account_id 最多一条规范逾期记录。", "overdue_records", "account_id", "一个账户出现多条逾期行。", "核对分页续接或重复块，确认唯一规范逾期行。", "message 中 dataset/record_id/fields；dataset_id、section_id；page_range 合并相关页", "否"],
  ["非信贷 / 公共记录", "章节封装", "章节状态", "PERSONAL_BRIEF_AUDIT_SECTION_STATUS_CONFLICT", "章节状态冲突", "error", "section.dataset_refs、record_status 与所属公开数据集行数必须一致。", "sections[] 及所属数据集", "dataset_refs / record_status / row_count", "章节引用不符，no_records/absent 却有行，或 reported 却无行。", "按 section_id 核对章节存在性、数据集引用和行数。", "message 中 fields；section_id、page_range", "否"],
  ["公共记录", "强类型公共记录", "聚合守恒", "PERSONAL_BRIEF_AUDIT_PUBLIC_RECORD_CONSERVATION_CONFLICT", "公共记录聚合守恒冲突", "error", "semantic public_records 各类型计数必须等于四个强类型公开数据集行数。", "欠税、民事判决、强制执行、行政处罚", "record_type ↔ typed dataset row count", "公共记录语义聚合与公开明细存在遗漏、重复或类型误分。", "按 record_type 比对语义聚合和 typed projector。", "message 中 dataset；dataset_id、section_id", "否"],
  ["系统审计", "审计器运行", "Fail-open", "PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR", "审计器内部错误", "warning", "捕获观察性审计器自身异常；业务结果继续输出。", "无业务数据集", "审计器异常类型", "本次审计未完整执行，但提取结果保持原样。", "查看服务日志和异常栈，修复审计器后重跑该报告。", "仅 code / level / message", "否"],
];

const expectedAuditCodes = auditRows.map((row) => row[3]);
if (new Set(expectedAuditCodes).size !== expectedAuditCodes.length) {
  throw new Error("Audit warning dictionary contains duplicate codes.");
}
const auditSource = await fs.readFile(`${workspace}/docmirror/plugins/credit_report/personal_brief_native/audit.py`, "utf8");
const implementedCodes = [...new Set(auditSource.match(/PERSONAL_BRIEF_AUDIT_[A-Z_]+/g) ?? [])].sort();
const documentedCodes = [...expectedAuditCodes].sort();
if (JSON.stringify(implementedCodes) !== JSON.stringify(documentedCodes)) {
  throw new Error(`Audit-code coverage mismatch. Implemented=${implementedCodes.join(",")} Documented=${documentedCodes.join(",")}`);
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));

// Update the landing sheet without changing its established layout.
const instructions = workbook.worksheets.getItem("使用说明");
instructions.getRange("A1").values = [["PBOC Personal Brief Community JSON Business & Audit Dictionary｜个人简版征信业务与审计字典"]];
instructions.getRange("A2").values = [["范围：个人简版 Community JSON 的 datasets[].rows[].normalized 业务字段、闭集枚举/金额口径及 warnings[] 观察性审计告警。"]];
instructions.getRange("B10").values = [["2026-08-10"]];
instructions.getRange("A10:B10").copyTo(instructions.getRange("A11:B11"), "all");
instructions.getRange("A11:B11").values = [["审计告警代码数", 22]];
instructions.getRange("D4").values = [["业务审计与告警使用规则"]];
instructions.getRange("D5:D17").values = [
  ["1. 本工作簿解释 15 个公开业务数据集、154 个业务字段，以及个人简版 warnings[]；record_id、raw/source 等其他 Community 技术外壳不列入字段字典。"],
  ["2. 公开 columns 是固定规范字段；某行 normalized 缺少字段或值为 null 时，表示源文未报告或不适用，须结合相邻 *_status 字段判断。"],
  ["3. 数值 0 和布尔 false 都是有效业务事实，不能与缺失/null 混为一谈。"],
  ["4. 个人简版报告金额口径固定为 CNY / CNY_1 / precision 0；account_currency 仅代表账户原始计价币种。"],
  ["5. long_id 在 Community 中表现为 string，严禁转成数值；标记为敏感的十个字段对外展示应脱敏。"],
  ["6. overdue_records.account_id 外键关联 credit_accounts.account_id；逾期记录应位于对应信用卡/贷款/其他业务账户之前。"],
  ["7. document_type、business_type、reason、query_channel 等按源报告原文保留；payment_status 当前闭集仅为“正常/欠费”。"],
  ["8. 枚举必须按 Dataset + Field 解释；Community columns[].enum 给出当前允许值，出现合同外值属于错误。"],
  ["9. repayment_records、聚合 public_records、report_notes 被公开投影有意省略；公共记录审计四个强类型数据集。"],
  ["10. 观察性审计只追加 warnings[]，不会重写、删除或纠正任何 datasets 业务值。"],
  ["11. level=error 表示确定性合同/业务关系矛盾；level=warning 表示完整性、来源或审计器状态需要人工复核。"],
  ["12. 追踪顺序：code → message 中 dataset/record_id/fields → dataset_id/section_id → page_range 回查源 PDF。"],
  ["13. PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR 为 fail-open：审计器失败时保留提取结果并追加系统告警。"],
];
instructions.getRange("D5:D17").format = {
  fill: colors.white,
  font: { fontSize: 11, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  verticalAlignment: "center",
};
instructions.getRange("D15:D17").format.fill = colors.paleYellow;
for (const row of [15, 16, 17]) {
  instructions.getRange(`D${row}:H${row}`).merge();
}
instructions.getRange("D15:H17").format = {
  fill: colors.paleYellow,
  font: { fontSize: 11, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  verticalAlignment: "center",
  rowHeight: 54,
};
instructions.getRange("A11:B11").format = {
  fill: colors.white,
  font: { fontSize: 11, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
};
instructions.getRange("A11").format = {
  fill: "#F3F6F9",
  font: { bold: true, fontSize: 11, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
};

// Update business-field descriptions to the stabilized enum and money contracts.
const fieldSheet = workbook.worksheets.getItem("字段字典");
const fieldValues = fieldSheet.getRange("A5:N158").values;
for (const row of fieldValues) {
  const field = String(row[5] ?? "");
  const communityType = String(row[6] ?? "");
  if (field === "reporting_amount_unit") {
    row[9] = "CNY_1（人民币元）；个人简版公开唯一允许值";
    row[12] = "必须等于 CNY_1，并与报告头及 money 列 unit 一致；否则产生金额口径告警。";
  } else if (field === "reporting_amount_currency") {
    row[9] = "CNY；个人简版报告金额币种";
    row[12] = "必须与报告头 reporting_currency=CNY 一致；不可与 account_currency 混淆。";
  } else if (field === "reporting_amount_precision") {
    row[9] = "0；CNY_1 下金额按整数元表达";
    row[12] = "必须与 CNY_1 口径一致；不得由下游自行补小数位或缩放。";
  }
  if (communityType === "money") {
    row[9] = "非负整数十进制字符串；币种 CNY；单位 CNY_1";
    const existing = String(row[12] ?? "");
    if (!existing.includes("CNY_1")) {
      row[12] = `${existing}${existing ? "；" : ""}须通过金额值与金额口径审计（CNY_1）。`;
    }
  }
  if (field === "payment_status") {
    row[8] = "截至报告时点该项服务的缴费或欠费状态；当前个人简版闭集为“正常”或“欠费”。";
    row[9] = "封闭枚举：正常 / 欠费；见“枚举值字典”";
    row[12] = "出现其他值将违反 dataset-qualified 枚举合同。";
  }
  if (field === "account_lifecycle_state") {
    row[12] = "允许 open / settled / closed / transferred_out；unknown 不得进入公开 Community JSON。";
  }
  if (field === "payoff_state") {
    row[12] = "unknown 仅允许用于 transferred_out 账户；其他生命周期出现 unknown 属于关系冲突。";
  }
  if (field === "credit_quality_status") {
    row[8] = "源报告明确记载的特殊信贷质量状态；当前公开闭集仅含 bad_debt 与 not_reported。";
    row[12] = "normal / unresolved 不属于当前公开合同；若出现应由上游修复或阻断。";
  }
}
fieldSheet.getRange("A5:N158").values = fieldValues;

// Update the dataset index where the stabilized contract changes reviewer behavior.
const indexSheet = workbook.worksheets.getItem("数据集索引");
const indexValues = indexSheet.getRange("A5:K19").values;
for (const row of indexValues) {
  const dataset = String(row[4] ?? "");
  if (dataset === "personal_report_metadata") {
    row[10] = "核对报告编号/时间、主身份及 CNY/CNY_1/0 金额口径；marital_status_raw 用于复核规范化。";
  }
  if (dataset === "postpaid_records") {
    row[9] = 2;
    row[10] = "payment_status 当前仅允许“正常/欠费”；金额字段遵循 CNY/CNY_1。";
  }
}
indexSheet.getRange("A5:K19").values = indexValues;

// Stabilize the value dictionary while retaining rejected legacy values as audit references.
const valueSheet = workbook.worksheets.getItem("枚举值字典");
valueSheet.getRange("A2").values = [["英文规范值 → 中文业务含义；同时标明当前允许值、旧别名与禁止公开值，供 Community JSON 审计使用。"]];
const valueValues = valueSheet.getRange("A5:J120").values;
for (const row of valueValues) {
  const dataset = String(row[3] ?? "");
  const field = String(row[4] ?? "");
  const value = String(row[6] ?? "");
  if (field === "reporting_amount_unit" && value === "yuan") {
    row[7] = "元（旧内部别名）";
    row[8] = "不得公开（会规范化）";
    row[9] = "进入 Community 前必须规范为 CNY_1；公开 JSON 出现 yuan 视为金额口径冲突。";
  }
  if (field === "reporting_amount_unit" && value === "CNY_1") {
    row[7] = "元（人民币）";
    row[8] = "当前唯一允许值";
    row[9] = "个人简版 Community JSON 的报告金额单位。";
  }
  if (field === "reporting_amount_unit" && value === "CNY_10K") {
    row[8] = "不得公开";
    row[9] = "当前个人简版合同不允许；审核端不得自行做 10,000 倍缩放。";
  }
  if (dataset === "credit_accounts" && field === "account_lifecycle_state" && value === "unknown") {
    row[8] = "不得公开";
    row[9] = "当前闭合合同不允许 unknown；出现时应由上游修复或阻断。";
  }
  if (dataset === "credit_accounts" && field === "payoff_state" && value === "unknown") {
    row[8] = "条件允许";
    row[9] = "仅允许 account_lifecycle_state=transferred_out；其他组合触发账户关系告警。";
  }
  if (dataset === "credit_accounts" && field === "credit_quality_status" && value === "normal") {
    row[8] = "不得公开";
    row[9] = "不在当前公开枚举合同；若出现，视为枚举合同违例。";
  }
  if (dataset === "credit_accounts" && field === "credit_quality_status" && value === "unresolved") {
    row[8] = "不得公开";
    row[9] = "解析不确定性哨兵，不是业务值；必须在公开输出前解决。";
  }
}
valueSheet.getRange("A5:J120").values = valueValues;
const paymentValueRows = [
  ["非信贷交易记录", "后付费记录", "后付费非信贷交易记录", "postpaid_records", "payment_status", "当前缴费状态", "正常", "正常", "当前允许值", "源报告值；属于当前个人简版闭集。"],
  ["非信贷交易记录", "后付费记录", "后付费非信贷交易记录", "postpaid_records", "payment_status", "当前缴费状态", "欠费", "欠费", "当前允许值", "源报告值；属于当前个人简版闭集。"],
];
const existingPaymentRows = valueValues.filter((row) => row[3] === "postpaid_records" && row[4] === "payment_status");
if (existingPaymentRows.length === 0) {
  const valueTable = valueSheet.tables.getItem("PersonalBriefBusinessValues");
  valueTable.rows.add(null, paymentValueRows);
}
valueSheet.getRange("A121:J122").format = {
  font: { fontSize: 10, color: colors.text },
  borders: {
    insideHorizontal: { style: "thin", color: colors.border },
    bottom: { style: "thin", color: colors.border },
    left: { style: "thin", color: colors.border },
    right: { style: "thin", color: colors.border },
  },
  wrapText: true,
  verticalAlignment: "center",
};
valueSheet.getRange("A121:J121").format.fill = colors.paleYellow;
valueSheet.getRange("A122:J122").format.fill = colors.blueBand;

// Add the traceable audit-warning dictionary.
const auditSheet = workbook.worksheets.add("审计告警字典");
auditSheet.showGridLines = false;
auditSheet.getRange("A1:M1").merge();
auditSheet.getRange("A1").values = [["个人简版 Community JSON 审计告警字典｜Audit Warning Dictionary"]];
auditSheet.getRange("A1:M1").format = {
  fill: colors.navy,
  font: { bold: true, fontSize: 18, color: colors.white, typeface: "Carlito" },
  rowHeight: 30,
  verticalAlignment: "center",
};
auditSheet.getRange("A2:M2").merge();
auditSheet.getRange("A2").values = [["22 个 personal-brief-only 观察性审计代码；告警只追加到 warnings[]，不会改变任何提取业务值。"]];
auditSheet.getRange("A2:M2").format = {
  fill: colors.subtitle,
  font: { italic: true, fontSize: 11, color: colors.text, typeface: "Carlito" },
  wrapText: true,
  rowHeight: 28,
};
auditSheet.getRange("A3:M3").format.rowHeight = 8;

auditSheet.getRange("A4:C4").values = [["warnings[] 字段", "JSON 格式", "业务审计解释"]];
auditSheet.getRange("A5:C11").values = [
  ["code", "PERSONAL_BRIEF_AUDIT_*", "稳定规则代码；优先用它筛选和聚合。"],
  ["level", "error / warning", "error=确定性矛盾；warning=完整性/来源/审计器待复核。"],
  ["message", "字符串", "前缀可含 dataset、record_id、fields，随后是业务含义。"],
  ["dataset_id", "可选字符串", "定位到 Community datasets[].id。"],
  ["section_id", "可选字符串", "定位到 Community sections[].id。"],
  ["page_range", "可选 [起页,止页]", "定位源 PDF 页；来源不足或全局规则时可缺省。"],
  ["warnings=[]", "空数组", "表示未发现已实现的确定性矛盾，不等于逐字证明源数据正确。"],
];
auditSheet.getRange("A4:C4").format = {
  fill: colors.header,
  font: { bold: true, color: colors.white, fontSize: 11 },
  borders: { preset: "all", style: "thin", color: colors.border },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
auditSheet.getRange("A5:C11").format = {
  fill: colors.white,
  font: { fontSize: 10, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  verticalAlignment: "center",
};
auditSheet.getRange("E4:H4").merge();
auditSheet.getRange("E4").values = [["审计定位流程"]];
auditSheet.getRange("E4:H4").format = {
  fill: colors.header,
  font: { bold: true, color: colors.white, fontSize: 11 },
  borders: { preset: "all", style: "thin", color: colors.border },
  horizontalAlignment: "center",
};
const workflow = [
  "1. 先按 level 筛选：error 优先，warning 进入人工复核队列。",
  "2. 用 code 在下方规则字典确定检查逻辑和影响范围。",
  "3. 从 message 读取 dataset / record_id / fields，定位 normalized 行。",
  "4. 用 dataset_id、section_id 核对 Community 导航与数据集归属。",
  "5. 用 page_range 回查源 PDF；再结合字段字典与枚举值字典判断。",
  "6. 告警不自动纠正结果；复核后应在提取/重建逻辑中修复并重跑。",
  "7. INTERNAL_ERROR 表示审计器 fail-open，本次业务值仍按原提取结果输出。",
];
workflow.forEach((text, index) => {
  const row = 5 + index;
  auditSheet.getRange(`E${row}:H${row}`).merge();
  auditSheet.getRange(`E${row}`).values = [[text]];
});
auditSheet.getRange("E5:H11").format = {
  fill: colors.white,
  font: { fontSize: 10, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  verticalAlignment: "center",
};

auditSheet.getRange("J4:M4").merge();
auditSheet.getRange("J4").values = [["规则概览与可追踪 Warning 示例"]];
auditSheet.getRange("J4:M4").format = {
  fill: colors.header,
  font: { bold: true, color: colors.white, fontSize: 11 },
  borders: { preset: "all", style: "thin", color: colors.border },
  horizontalAlignment: "center",
};
auditSheet.getRange("J5:M6").values = [
  ["规则代码数", null, "error 代码数", null],
  ["warning 代码数", null, "修改提取值", "否"],
];
auditSheet.getRange("K5").formulas = [["=COUNTA(D15:D36)"]];
auditSheet.getRange("M5").formulas = [["=COUNTIF(F15:F36,\"error\")"]];
auditSheet.getRange("K6").formulas = [["=COUNTIF(F15:F36,\"warning\")"]];
auditSheet.getRange("J5:M6").format = {
  fill: colors.white,
  font: { fontSize: 10, color: colors.text },
  borders: { preset: "all", style: "thin", color: colors.border },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
auditSheet.getRange("J5:J6").format.font = { bold: true, fontSize: 10, color: colors.text };
auditSheet.getRange("L5:L6").format.font = { bold: true, fontSize: 10, color: colors.text };
auditSheet.getRange("J8:M8").merge();
auditSheet.getRange("J8").values = [["示例（合成告警，仅演示追踪字段）"]];
auditSheet.getRange("J8:M8").format = {
  fill: colors.paleYellow,
  font: { bold: true, fontSize: 10, color: colors.amberText },
  borders: { preset: "all", style: "thin", color: colors.border },
  horizontalAlignment: "center",
};
auditSheet.getRange("J9:M11").merge();
auditSheet.getRange("J9").values = [["{\n  \"code\": \"PERSONAL_BRIEF_AUDIT_AMOUNT_STATUS_CONFLICT\",\n  \"level\": \"error\",\n  \"message\": \"[dataset=credit_accounts; record_id=credit_account:1; fields=balance,balance_status] Amount presence disagrees with its reporting-status field.\",\n  \"dataset_id\": \"ds_credit_accounts\", \"section_id\": \"sec_credit\", \"page_range\": [1,2]\n}"]];
auditSheet.getRange("J9:M11").format = {
  fill: "#F7F9FB",
  font: { fontSize: 9, color: colors.text, typeface: "Consolas" },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  verticalAlignment: "top",
};

auditSheet.getRange("A13:M13").merge();
auditSheet.getRange("A13").values = [["规则按全局 → 报告信息 → 信贷记录 → 非信贷/公共记录 → 系统审计排列；同一 code 的多个触发分支合并为一条字典项。"]];
auditSheet.getRange("A13:M13").format = {
  fill: colors.subtitle,
  font: { italic: true, fontSize: 10, color: colors.text },
  wrapText: true,
  rowHeight: 24,
};
const auditHeaders = ["报告大章", "业务对象/范围", "审计类别", "Warning Code", "中文名称", "Level", "检查内容", "主要 Dataset", "主要字段/关系", "告警表示", "建议复核动作", "Community 定位字段", "修改提取值"];
auditSheet.getRange("A14:M14").values = [auditHeaders];
auditSheet.getRange("A15:M36").values = auditRows;
const auditTable = auditSheet.tables.add("A14:M36", true, "PersonalBriefAuditWarnings");
auditTable.style = "TableStyleMedium2";
auditSheet.getRange("A14:M14").format = {
  fill: colors.header,
  font: { bold: true, color: colors.white, fontSize: 10 },
  borders: { preset: "all", style: "thin", color: colors.border },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 36,
};
auditSheet.getRange("A15:M36").format = {
  font: { fontSize: 9, color: colors.text },
  borders: {
    insideHorizontal: { style: "thin", color: colors.border },
    bottom: { style: "thin", color: colors.border },
    left: { style: "thin", color: colors.border },
    right: { style: "thin", color: colors.border },
  },
  wrapText: true,
  verticalAlignment: "center",
  rowHeight: 72,
};
const groupFills = {
  "全局规则": colors.subtitle,
  "报告信息": "#DDEBF7",
  "信贷记录": colors.paleGreen,
  "信贷记录 / 公共记录": colors.paleGreen,
  "非信贷 / 公共记录": colors.paleOrange,
  "公共记录": colors.paleOrange,
  "系统审计": colors.palePurple,
};
auditRows.forEach((row, index) => {
  const excelRow = 15 + index;
  auditSheet.getRange(`A${excelRow}:C${excelRow}`).format.fill = groupFills[row[0]] ?? colors.white;
  auditSheet.getRange(`D${excelRow}:M${excelRow}`).format.fill = index % 2 === 0 ? colors.white : colors.paleBlue;
  if (row[5] === "error") {
    auditSheet.getRange(`F${excelRow}`).format = {
      fill: colors.paleRed,
      font: { bold: true, color: colors.redText, fontSize: 9 },
      horizontalAlignment: "center",
      verticalAlignment: "center",
    };
  } else {
    auditSheet.getRange(`F${excelRow}`).format = {
      fill: colors.paleYellow,
      font: { bold: true, color: colors.amberText, fontSize: 9 },
      horizontalAlignment: "center",
      verticalAlignment: "center",
    };
  }
  auditSheet.getRange(`M${excelRow}`).format = {
    fill: colors.paleGreen,
    font: { bold: true, color: "#375623", fontSize: 9 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
});

const widths = {
  A: 16, B: 22, C: 18, D: 54, E: 22, F: 10, G: 50, H: 38, I: 46, J: 48, K: 48, L: 44, M: 14,
};
for (const [column, width] of Object.entries(widths)) {
  auditSheet.getRange(`${column}1:${column}36`).format.columnWidth = width;
}
auditSheet.getRange("A4:M11").format.rowHeight = 32;
auditSheet.getRange("J9:M11").format.rowHeight = 44;
auditSheet.freezePanes.freezeRows(14);
auditSheet.freezePanes.freezeColumns(4);

// Workbook-wide focused usability updates.
instructions.showGridLines = false;
indexSheet.showGridLines = false;
fieldSheet.showGridLines = false;
valueSheet.showGridLines = false;
instructions.freezePanes.freezeRows(4);
indexSheet.freezePanes.freezeRows(4);
fieldSheet.freezePanes.freezeRows(4);
fieldSheet.freezePanes.freezeColumns(4);
valueSheet.freezePanes.freezeRows(4);
valueSheet.freezePanes.freezeColumns(4);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

// Re-import the saved workbook, render every sheet, and run structural/formula checks.
const verified = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const formulaErrors = [];
for (const sheet of verified.worksheets.items) {
  const used = sheet.getUsedRange();
  const values = used?.values ?? [];
  values.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
    if (typeof value === "string" && /^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/.test(value)) {
      formulaErrors.push(`${sheet.name}!R${rowIndex + 1}C${colIndex + 1}:${value}`);
    }
  }));
  const rendered = await verified.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheet.name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await rendered.arrayBuffer()));
}
if (formulaErrors.length) {
  throw new Error(`Formula errors found: ${formulaErrors.join(", ")}`);
}
const auditCheck = verified.worksheets.getItem("审计告警字典").getRange("J5:M6").values;
const enumUsed = verified.worksheets.getItem("枚举值字典").getUsedRange().address;
const auditUsed = verified.worksheets.getItem("审计告警字典").getUsedRange().address;
process.stdout.write(JSON.stringify({
  outputPath,
  sheets: verified.worksheets.items.map((sheet) => ({ name: sheet.name, used: sheet.getUsedRange().address })),
  auditCheck,
  implementedAuditCodes: implementedCodes.length,
  documentedAuditCodes: documentedCodes.length,
  enumUsed,
  auditUsed,
  formulaErrors,
}, null, 2));

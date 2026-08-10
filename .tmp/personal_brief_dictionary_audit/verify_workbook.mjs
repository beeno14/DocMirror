import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workspace = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full";
const workbookPath = `${workspace}/outputs/personal_brief_community_dictionary_audit_20260810/PBOC_personal_brief_community_json_business_and_audit_dictionary_中英.xlsx`;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));

const requiredSheets = ["使用说明", "数据集索引", "字段字典", "枚举值字典", "审计告警字典"];
const actualSheets = workbook.worksheets.items.map((sheet) => sheet.name);
if (JSON.stringify(requiredSheets) !== JSON.stringify(actualSheets)) throw new Error(`Sheet mismatch: ${actualSheets}`);

const datasetRows = workbook.worksheets.getItem("数据集索引").getRange("A5:K19").values;
const fieldRows = workbook.worksheets.getItem("字段字典").getRange("A5:N158").values;
const valueRows = workbook.worksheets.getItem("枚举值字典").getRange("A5:J122").values;
const auditRows = workbook.worksheets.getItem("审计告警字典").getRange("A15:M36").values;
if (datasetRows.length !== 15) throw new Error(`Expected 15 datasets, found ${datasetRows.length}`);
if (fieldRows.length !== 154) throw new Error(`Expected 154 fields, found ${fieldRows.length}`);
if (auditRows.length !== 22) throw new Error(`Expected 22 audit codes, found ${auditRows.length}`);

const source = await fs.readFile(`${workspace}/docmirror/plugins/credit_report/personal_brief_native/audit.py`, "utf8");
const implemented = [...new Set(source.match(/PERSONAL_BRIEF_AUDIT_[A-Z_]+/g) ?? [])].sort();
const documented = auditRows.map((row) => String(row[3] ?? "")).sort();
if (JSON.stringify(implemented) !== JSON.stringify(documented)) throw new Error("Audit-code coverage differs from audit.py");
const errors = auditRows.filter((row) => row[5] === "error").length;
const warnings = auditRows.filter((row) => row[5] === "warning").length;
if (errors !== 18 || warnings !== 4) throw new Error(`Wrong severity totals: ${errors}/${warnings}`);
if (auditRows.some((row) => row[12] !== "否")) throw new Error("An audit rule incorrectly claims to mutate extraction values.");

const publicUnit = valueRows.find((row) => row[4] === "reporting_amount_unit" && row[6] === "CNY_1");
const legacyUnit = valueRows.find((row) => row[4] === "reporting_amount_unit" && row[6] === "yuan");
const forbiddenScale = valueRows.find((row) => row[4] === "reporting_amount_unit" && row[6] === "CNY_10K");
if (publicUnit?.[8] !== "当前唯一允许值") throw new Error("CNY_1 is not marked as the only public unit.");
if (!String(legacyUnit?.[8] ?? "").startsWith("不得公开")) throw new Error("yuan is not marked non-public.");
if (forbiddenScale?.[8] !== "不得公开") throw new Error("CNY_10K is not marked non-public.");
const paymentValues = valueRows.filter((row) => row[3] === "postpaid_records" && row[4] === "payment_status").map((row) => row[6]).sort();
if (JSON.stringify(paymentValues) !== JSON.stringify(["欠费", "正常"].sort())) throw new Error(`Payment-status values mismatch: ${paymentValues}`);

const formulaErrors = [];
for (const sheet of workbook.worksheets.items) {
  const values = sheet.getUsedRange().values;
  values.forEach((row, r) => row.forEach((value, c) => {
    if (typeof value === "string" && /^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/.test(value)) formulaErrors.push(`${sheet.name}!R${r + 1}C${c + 1}`);
  }));
}
if (formulaErrors.length) throw new Error(`Formula errors: ${formulaErrors.join(",")}`);

const stats = await fs.stat(workbookPath);
process.stdout.write(JSON.stringify({ workbookPath, bytes: stats.size, datasets: 15, businessFields: 154, valueRows: valueRows.length, auditCodes: 22, errorCodes: 18, warningCodes: 4, formulaErrors }, null, 2));

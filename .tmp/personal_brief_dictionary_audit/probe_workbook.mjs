import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full/outputs/personal_brief_business_dictionary_20260808/PBOC_personal_brief_community_json_business_dictionary_中英.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));

for (const [sheetName, rangeAddress, needles] of [
  ["字段字典", "A1:N158", ["reporting_amount_unit", "reporting_amount_currency", "reporting_amount_precision", "payment_status", "credit_quality_status", "account_lifecycle_state"]],
  ["枚举值字典", "A1:J120", ["yuan", "CNY_1", "CNY_10K", "normal", "unresolved", "unknown", "payment_status"]],
]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getRange(rangeAddress).values;
  process.stdout.write(`\n${sheetName}\n`);
  values.forEach((row, index) => {
    const text = row.map((value) => String(value ?? "")).join(" | ");
    if (needles.some((needle) => text.includes(needle))) {
      process.stdout.write(`${index + 1}: ${JSON.stringify(row)}\n`);
    }
  });
}

for (const sheet of workbook.worksheets.items) {
  process.stdout.write(`TABLES ${sheet.name}: ${sheet.tables.items.map((table) => `${table.name}:${table.getRange?.().address ?? "?"}`).join(", ")}\n`);
}

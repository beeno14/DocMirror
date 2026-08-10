import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full/outputs/personal_brief_business_dictionary_20260808/PBOC_personal_brief_community_json_business_dictionary_中英.xlsx";
const outDir = "D:/beeno/ValueMap/docmirror3.0-full/DocMirror-Full/.tmp/personal_brief_dictionary_audit/previews_before";
await fs.mkdir(outDir, { recursive: true });

const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 8,
  tableMaxCols: 12,
  tableMaxCellChars: 100,
});
process.stdout.write(`SUMMARY\n${summary.ndjson ?? summary}\n`);

const sheets = workbook.worksheets.items;
for (const sheet of sheets) {
  const used = sheet.getUsedRange();
  const region = await workbook.inspect({
    kind: "region",
    sheetId: sheet.name,
    range: used?.address ?? "A1:Z40",
    maxChars: 9000,
    tableMaxRows: 20,
    tableMaxCols: 16,
    tableMaxCellChars: 120,
  });
  process.stdout.write(`\nSHEET ${sheet.name}\nUSED ${used?.address ?? "unknown"}\n${region.ndjson ?? region}\n`);
  const style = await workbook.inspect({
    kind: "computedStyle",
    sheetId: sheet.name,
    range: used?.address ?? "A1:H12",
    maxChars: 5000,
  });
  process.stdout.write(`STYLE ${sheet.name}\n${style.ndjson ?? style}\n`);
  const rendered = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheet.name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(outDir, `${safeName}.png`), new Uint8Array(await rendered.arrayBuffer()));
}

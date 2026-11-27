#requires -Version 7.0
param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
# Always keep the builder output separate from the final native Word DOCX.
$ErrorActionPreference='Stop'
$inputFull=(Resolve-Path -LiteralPath $InputPath).Path
$outputFull=[IO.Path]::GetFullPath($OutputPath)
if($inputFull -eq $outputFull) {throw 'Use a separate output path.'}
$wordApp=New-Object -ComObject Word.Application
$wordApp.Visible=$false
$wordApp.DisplayAlerts=0
$wordApp.AutomationSecurity=3
$wordDoc=$null
try {
    $wordDoc=$wordApp.Documents.Open($inputFull,$false,$true)
    $wordDoc.Fields.Update() | Out-Null
    foreach($toc in $wordDoc.TablesOfContents) {$toc.Update()}
    $wordDoc.Repaginate()
    foreach($toc in $wordDoc.TablesOfContents) {$toc.UpdatePageNumbers()}
    $wordDoc.SaveAs2($outputFull,16)
    Write-Output ('Saved Word-native report: '+$outputFull)
} finally {
    if($null -ne $wordDoc) {$wordDoc.Close(0)}
    $wordApp.Quit()
    if($null -ne $wordDoc) {[Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc) | Out-Null}
    [Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}

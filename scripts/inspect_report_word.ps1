#requires -Version 7.0
param(
    [Parameter(Mandatory=$true)][string]$DocumentPath,
    [Parameter(Mandatory=$true)][string]$QaDirectory
)
# Fallback QA export for a final DOCX already saved/refreshed in Microsoft Word.
$ErrorActionPreference='Stop'
$sourcePath=(Resolve-Path -LiteralPath $DocumentPath).Path
$qaPath=[IO.Path]::GetFullPath($QaDirectory)
New-Item -ItemType Directory -Path $qaPath -Force | Out-Null
$inspectionPath=Join-Path $qaPath 'input.docx'
Copy-Item -LiteralPath $sourcePath -Destination $inspectionPath -Force
$wordApp=New-Object -ComObject Word.Application
$wordApp.Visible=$false
$wordApp.DisplayAlerts=0
$wordApp.AutomationSecurity=3
$wordDoc=$null
try {
    $wordDoc=$wordApp.Documents.Open($inspectionPath,$false,$true)
    $wordDoc.ExportAsFixedFormat((Join-Path $qaPath 'report.pdf'),17,$false,0,0,1,1,0,$true,$false,0,$false,$true,$false)
    $pageCount=$wordDoc.ComputeStatistics(2)
    $wordDoc.Bookmarks.ShowHidden=$true
    $pageMap=@{}
    foreach($bookmark in $wordDoc.Bookmarks) {
        if($bookmark.Name -match '^([hft][0-9]+|_Toc[0-9]+)$') {
            $pageMap[$bookmark.Name]=@{page=$bookmark.Range.Information(1);section=$bookmark.Range.Information(2);text=$bookmark.Range.Text.Trim()}
        }
    }
    $pageMap | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $qaPath 'pages.json') -Encoding utf8
    Write-Output ('Pages: '+$pageCount)
    Write-Output ('Bookmarks: '+$pageMap.Count)
} finally {
    if($null -ne $wordDoc) {$wordDoc.Close(0)}
    $wordApp.Quit()
    if($null -ne $wordDoc) {[Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc) | Out-Null}
    [Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}

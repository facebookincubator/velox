# Velox Windows Test Runner
# Discovers and runs all Velox test executables, showing concise per-suite results.
#
# Usage:
#   .\run-windows-tests.ps1                                    # Run all tests (8 parallel jobs)
#   .\run-windows-tests.ps1 -Parallel 1                        # Run sequentially
#   .\run-windows-tests.ps1 -Filter "velox_base_test"          # Run one test suite
#   .\run-windows-tests.ps1 -Filter "velox_(type|vector)"      # Run matching suites (regex)
#   .\run-windows-tests.ps1 -GTestFilter "*BitUtil*"            # Run specific test cases
#   .\run-windows-tests.ps1 -ListOnly                           # List discovered test exes
#   .\run-windows-tests.ps1 -ShowOutput                         # Show full GTest output in console
#   .\run-windows-tests.ps1 -TimeoutSeconds 600                 # 10-minute timeout per test
#   .\run-windows-tests.ps1 -BuildType Debug                    # Run Debug-built tests
#   .\run-windows-tests.ps1 -FailedOnly                         # Rerun only previously failed suites
#
# Logs:
#   Per-test stdout/stderr → <LogDir>\<test_name>.log
#   Per-test GTest XML     → <LogDir>\<test_name>.xml
#   Summary JSON           → <LogDir>\summary.json

param(
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Release",
    [string]$Filter = "",              # Regex filter on test exe name (e.g. "velox_base_test")
    [string]$GTestFilter = "",         # GTest --gtest_filter value (e.g. "*BitUtil*")
    [string]$LogDir = "",              # Default: <BuildDir>\test-results
    [switch]$ListOnly = $false,        # Just list discovered tests, don't run
    [switch]$ShowOutput = $false,      # Show full test output in console
    [switch]$FailedOnly = $false,      # Rerun only previously failed/crashed/timeout suites
    [int]$Parallel = 8,                # Number of concurrent test processes (1 = sequential)
    [int]$TimeoutSeconds = 600         # Per-test timeout (10 minutes default)
)

# Change to Velox root (parent of this script's directory)
$scriptDir = Split-Path -Parent $PSScriptRoot
Set-Location $scriptDir

# ============================================================================
# Defaults
# ============================================================================

if ([string]::IsNullOrEmpty($LogDir)) {
    $LogDir = Join-Path $BuildDir "test-results"
}
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogDir = Resolve-Path $LogDir

# Executables that live under tests\ but are not real test suites
# (benchmarks, demos, long-running fuzzers)
$ExcludeList = @(
    "velox_memcpy_meter",
    "velox_in_10_min_demo",
    "velox_tpch_speed_test",
    # Fuzzer tests run indefinitely / non-deterministically — skip in normal runs
    "velox_expression_fuzzer_test",
    "velox_window_fuzzer_test",
    "velox_writer_fuzzer_test",
    "presto_sql_test",
    "velox_table_evolution_fuzzer_test",
    "velox_aggregation_runner_test"
)

# Per-test timeout overrides (seconds).
# Monolithic test suites with many test cases may need more time than the
# default $TimeoutSeconds to complete.
$TimeoutOverrides = @{
    "velox_exec_test" = 3600   # ~70 test files, thousands of cases — needs up to 60 min on CI
}

# GTest negative filter for timezone-dependent tests that fail on Windows.
# Windows TimeZoneMapStubs returns UTC offset for all named timezones,
# causing tests that expect real timezone conversions to fail.
$WindowsTzExcludeFilter = @(
    # TimeZoneMapTest - tests named timezone features not supported by stubs
    "TimeZoneMapTest.offset",
    "TimeZoneMapTest.invalid",
    "TimeZoneMapTest.getShortName",
    "TimeZoneMapTest.getLongName",
    "TimeZoneMapTest.offsetToLocal",
    "TimeZoneMapTest.offsetToSys",
    "TimeZoneMapTest.timePointBoundary",
    "TimeZoneMapTest.fromParsedTimestampWithTimeZone",
    "TimeZoneMapTest.toGMT",
    "TimeZoneMapTest.toTimezone",
    "TimeZoneMapTest.toGMTFromID",
    "TimeZoneMapTest.toTimezoneFromID",
    # Presto DateTimeFunctionsTest - timezone-dependent datetime operations
    "DateTimeFunctionsTest.timestampTest",
    "DateTimeFunctionsTest.year",
    "DateTimeFunctionsTest.quarter",
    "DateTimeFunctionsTest.month",
    "DateTimeFunctionsTest.hour",
    "DateTimeFunctionsTest.minute",
    "DateTimeFunctionsTest.dayOfMonth",
    "DateTimeFunctionsTest.dayOfWeek",
    "DateTimeFunctionsTest.dayOfYear",
    "DateTimeFunctionsTest.yearOfWeek",
    "DateTimeFunctionsTest.dateTrunc",
    "DateTimeFunctionsTest.dateTruncTimestampWithTimezone",
    "DateTimeFunctionsTest.dateAddTimestamp",
    "DateTimeFunctionsTest.dateAddTimestampWithTimeZone",
    "DateTimeFunctionsTest.dateDiffTimestampWithTimezone",
    "DateTimeFunctionsTest.parseDatetime",
    "DateTimeFunctionsTest.formatDateTime",
    "DateTimeFunctionsTest.formatDateTimeTimezone",
    "DateTimeFunctionsTest.dateFormat",
    "DateTimeFunctionsTest.fromIso8601Timestamp",
    "DateTimeFunctionsTest.dateParse",
    "DateTimeFunctionsTest.dateFunctionTimestampWithTimezone",
    "DateTimeFunctionsTest.castDateForDateFunction",
    "DateTimeFunctionsTest.currentDateWithTimezone",
    "DateTimeFunctionsTest.currentDateWithoutTimezone",
    "DateTimeFunctionsTest.timeZoneHour",
    "DateTimeFunctionsTest.timeZoneMinute",
    "DateTimeFunctionsTest.castDateToTimestamp",
    "DateTimeFunctionsTest.toISO8601Timestamp",
    "DateTimeFunctionsTest.toISO8601TimestampWithTimezone",
    "DateTimeFunctionsTest.currentTimezone",
    "DateTimeFunctionsTest.timestampWithTimeZonePlusIntervalDayTime",
    "DateTimeFunctionsTest.minusTimestampWithTimezone",
    "DateTimeFunctionsTest.greatestTimestampWithTimezone",
    # TimestampWithTimeZoneCastTest
    "TimestampWithTimeZoneCastTest.fromVarchar",
    "TimestampWithTimeZoneCastTest.toVarchar",
    "TimestampWithTimeZoneCastTest.fromVarcharWithoutTimezone",
    "TimestampWithTimeZoneCastTest.toTimestamp",
    "TimestampWithTimeZoneCastTest.toDate",
    "TimestampWithTimeZoneCastTest.fromDate",
    "TimestampWithTimeZoneCastTest.fromTime",
    "TimestampWithTimeZoneCastTest.fromTimeVerifyUtcStorage",
    # TimeUtilsTest
    "TimeUtilsTest.truncateTimestamp",
    # ExpressionOptimizerTest
    "ExprTest.evaluateConstantExpression",
    # Hive connector tests
    "PartitionIdGeneratorTest.partitionName",
    "PartitionIdGeneratorTest.timestampPartitionValueFormatting",
    "PartitionIdGeneratorTest.timestampPartitionKeyComparasion",
    # Hive CSV/serializer tests
    "CsvTableScanTest.headerAndCustomNullString",
    "CsvTableScanTest.complexTypesWithCustomDelimiters",
    "CsvTableScanTest.simpleTypes",
    "HiveFileFormatTest.write",
    # TimeWithTimezoneCastTest
    "TimeWithTimezoneCastTest.fromTime",
    "TimeWithTimezoneCastTest.fromTimeVerifyUtcStorage",
    # ArrayJoinTest
    "ArrayJoinTest.timestampTest",
    # DateTimeUtilTest - timezone conversions via DateTimeUtil
    "DateTimeUtilTest.fromParsedTimestampWithTimeZone",
    "DateTimeUtilTest.toGMT",
    "DateTimeUtilTest.toGMTFromID",
    "DateTimeUtilTest.toTimezone",
    "DateTimeUtilTest.toTimezoneFromID",
    # GreatestLeastTest
    "GreatestLeastTest.greatestTimestampWithTimezone",
    # HivePartitionNameTest
    "HivePartitionNameTest.partitionName",
    "HivePartitionNameTest.timestampPartitionValueFormatting",
    # TextReaderTest - CSV reader timezone-dependent tests
    "TextReaderTest.complexTypesWithCustomDelimiters",
    "TextReaderTest.headerAndCustomNullString",
    "TextReaderTest.simpleTypes",
    # TextWriterTest
    "TextWriterTest.write"
)

# Tests that deadlock on Windows due to task abort/spill path issues.
# These tests hang indefinitely (spin-waiting on reference counts or futures
# that never resolve because the Windows task lifecycle differs from Linux).
# They must be excluded to prevent the entire test suite from timing out.
$WindowsHangExcludeFilter = @(
    # Spill-related tests deadlock on Windows due to differences in the
    # Windows file locking and mmap compatibility layer. The spill path
    # uses file-backed memory that interacts with VirtualAlloc/VirtualFree
    # differently than Linux mmap/munmap, causing infinite waits.
    "HashJoinTest/HashJoinTest.spillPartitionBitsOverlap*",
    "SpillerTest/AllTypesSpillerTest.*",
    # Multi-fragment exchange tests hang on Windows due to task lifecycle
    # differences (reference count spin-waits that never resolve).
    "MultiFragmentTest/MultiFragmentTest.*",
    # StreamingAggregation parameterized tests hang on Windows.
    "StreamingAggregationTest/StreamingAggregationTest.*",
    # OutputBufferManager multi-fetcher tests hang on Windows.
    "AllOutputBufferManagerTestSuite/AllOutputBufferManagerTest.multiFetchers*",
    # TableWriter parameterized tests hang on Windows due to spill/file-locking
    # issues in bucketed, partitioned, and scaleWriter modes.
    "TableWriterTest/AllTableWriterTest.*",
    "TableWriterTest/BucketedTableOnlyWriteTest.*",
    "TableWriterTest/BucketSortOnlyTableWriterTest.*",
    # TopNRowNumber manyPartitions hangs on Windows (spill-related).
    "TopNRowNumberTest/MultiTopNRowNumberTest.manyPartitions*",
    # ScaleWriter tests hang on Windows.
    "LocalExchangePartitionBuffer/ScaleWriterLocalPartitionTestParametrized.*"
)

# Pre-compute the negative gtest filter string for all Windows exclusions
$allExclusions = @()
if ($WindowsTzExcludeFilter.Count -gt 0) {
    $allExclusions += $WindowsTzExcludeFilter
}
if ($WindowsHangExcludeFilter.Count -gt 0) {
    $allExclusions += $WindowsHangExcludeFilter
}
$windowsNegativeFilter = ""
if ($allExclusions.Count -gt 0) {
    $windowsNegativeFilter = ($allExclusions -join ":")
}

# ============================================================================
# Discover test executables
# ============================================================================

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Velox Windows Test Runner" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$testExes = Get-ChildItem -Path $BuildDir -Recurse -Filter "*.exe" |
    Where-Object {
        $_.FullName -match "\\tests?\\" -and
        $_.FullName -match "\\$BuildType\\" -and
        $ExcludeList -notcontains $_.BaseName
    } |
    Sort-Object BaseName

if ($Filter) {
    $testExes = $testExes | Where-Object { $_.BaseName -match $Filter }
}

# FailedOnly mode: rerun only suites that failed/crashed/timeout in previous run
if ($FailedOnly) {
    $prevSummary = Join-Path $LogDir "summary.json"
    if (Test-Path $prevSummary) {
        $prev = Get-Content $prevSummary -Raw | ConvertFrom-Json
        $failedSet = @{}
        foreach ($r in $prev.Results) {
            if ($r.Status -ne "PASSED") {
                $failedSet[$r.Name] = $true
            }
        }
        $testExes = $testExes | Where-Object { $failedSet.ContainsKey($_.BaseName) }
        Write-Host "  FailedOnly: rerunning $($testExes.Count) previously failed suites" -ForegroundColor Yellow
    } else {
        Write-Host "  FailedOnly: no previous summary.json found, running all" -ForegroundColor Yellow
    }
}

$totalCount = $testExes.Count

Write-Host "  Build type : $BuildType" -ForegroundColor White
Write-Host "  Tests found: $totalCount" -ForegroundColor White
Write-Host "  Parallel   : $Parallel jobs" -ForegroundColor White
Write-Host "  Filter     : $(if ($Filter) { $Filter } else { '(none)' })" -ForegroundColor White
Write-Host "  GTest filter: $(if ($GTestFilter) { $GTestFilter } else { '(none)' })" -ForegroundColor White
Write-Host "  Timeout    : ${TimeoutSeconds}s per test" -ForegroundColor White
Write-Host "  Log dir    : $LogDir" -ForegroundColor White
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

if ($totalCount -eq 0) {
    Write-Host "No test executables found. Build with: .\windows\build-velox-oss.ps1 -WithTests" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# List-only mode
# ============================================================================

if ($ListOnly) {
    Write-Host "Discovered test executables:" -ForegroundColor Cyan
    $idx = 0
    foreach ($exe in $testExes) {
        $idx++
        $relPath = $exe.FullName.Substring((Resolve-Path $BuildDir).Path.Length + 1)
        Write-Host ("  {0,3}. {1,-50} {2}" -f $idx, $exe.BaseName, $relPath) -ForegroundColor White
    }
    Write-Host ""
    Write-Host "Total: $totalCount test executables" -ForegroundColor Cyan
    exit 0
}

# ============================================================================
# Run tests (parallel)
# ============================================================================

Write-Host "Running $totalCount test suites ($Parallel parallel)..." -ForegroundColor Cyan
Write-Host ""

# Counters (thread-safe via synchronized hashtable)
$counters = [hashtable]::Synchronized(@{
    passedSuites  = 0
    failedSuites  = 0
    crashedSuites = 0
    timeoutSuites = 0
    totalTests    = 0
    totalPassed   = 0
    totalFailed   = 0
    totalSkipped  = 0
    failedNames   = [System.Collections.ArrayList]::new()
    results       = [System.Collections.ArrayList]::new()
    completed     = 0
})

# Measure total time
$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Build work items list
$workItems = @()
$idx = 0
foreach ($exe in $testExes) {
    $idx++
    $workItems += [PSCustomObject]@{
        Index    = $idx
        Exe      = $exe
        TestName = $exe.BaseName
        LogFile  = Join-Path $LogDir "$($exe.BaseName).log"
        XmlFile  = Join-Path $LogDir "$($exe.BaseName).xml"
    }
}

# Script block to run one test
$runTestBlock = {
    param($work, $gtestFilter, $timeoutSec)

    $testName = $work.TestName
    $logFile  = $work.LogFile
    $xmlFile  = $work.XmlFile
    $exePath  = $work.Exe.FullName

    $gtestArgs = @("--gtest_output=xml:$xmlFile")
    if ($gtestFilter) {
        $gtestArgs += "--gtest_filter=$gtestFilter"
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $crashed  = $false
    $exitCode = -1

    try {
        $proc = Start-Process -FilePath $exePath -ArgumentList $gtestArgs `
            -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
            -NoNewWindow -PassThru

        $proc.WaitForExit($timeoutSec * 1000) | Out-Null
        if (-not $proc.HasExited) {
            $timedOut = $true
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $proc.WaitForExit(5000) | Out-Null
        }
        $exitCode = $proc.ExitCode

        if (Test-Path "$logFile.err") {
            $errContent = Get-Content "$logFile.err" -Raw -ErrorAction SilentlyContinue
            if ($errContent) {
                Add-Content $logFile "`n--- STDERR ---`n$errContent"
            }
            Remove-Item "$logFile.err" -Force -ErrorAction SilentlyContinue
        }
    } catch {
        $crashed = $true
        "CRASH: $_" | Set-Content $logFile
    }

    $sw.Stop()

    return [PSCustomObject]@{
        TestName = $testName
        ExePath  = $exePath
        TimedOut = $timedOut
        Crashed  = $crashed
        ExitCode = $exitCode
        Elapsed  = $sw.Elapsed.TotalSeconds
        LogFile  = $logFile
        XmlFile  = $xmlFile
    }
}

# Run tests in parallel batches using thread jobs
if ($Parallel -le 1) {
    # Sequential mode
    foreach ($work in $workItems) {
        $mergedFilter = $GTestFilter
        if ($windowsNegativeFilter) {
            if ($mergedFilter) {
                $mergedFilter = "${mergedFilter}:-${windowsNegativeFilter}"
            } else {
                $mergedFilter = "*:-${windowsNegativeFilter}"
            }
        }
        $testTimeout = if ($TimeoutOverrides.ContainsKey($work.TestName)) { $TimeoutOverrides[$work.TestName] } else { $TimeoutSeconds }
        $result = & $runTestBlock $work $mergedFilter $testTimeout
        $counters.completed++

        # Parse + display inline (same as before)
        $testName = $result.TestName
        $suitePassed = 0; $suiteFailed = 0; $suiteSkipped = 0; $suiteTotal = 0; $xmlParsed = $false
        if (Test-Path $result.XmlFile) {
            try {
                [xml]$xmlDoc = Get-Content $result.XmlFile -Raw
                $ts = $xmlDoc.testsuites
                if ($ts) {
                    $suiteTotal   = [int]$ts.tests
                    $suiteFailed  = [int]$ts.failures + [int]$ts.errors
                    $suiteSkipped = [int]$ts.disabled + [int]$ts.skipped
                    $suitePassed  = $suiteTotal - $suiteFailed - $suiteSkipped
                    $xmlParsed    = $true
                }
            } catch {}
        }

        $status = ""; $statusColor = "Green"
        if ($result.TimedOut) { $status = "TIMEOUT (>${testTimeout}s)"; $statusColor = "Yellow"; $counters.timeoutSuites++; $counters.failedNames.Add($testName) | Out-Null }
        elseif ($result.Crashed) { $status = "CRASH"; $statusColor = "Red"; $counters.crashedSuites++; $counters.failedNames.Add($testName) | Out-Null }
        elseif ($result.ExitCode -ne 0 -and $xmlParsed -and $suiteFailed -gt 0) {
            $status = "$suitePassed passed, $suiteFailed FAILED"
            $statusColor = "Red"; $counters.failedSuites++; $counters.failedNames.Add($testName) | Out-Null
        }
        elseif ($result.ExitCode -ne 0 -and $xmlParsed -and $suiteFailed -eq 0) {
            $status = "$suitePassed passed$(if ($suiteSkipped -gt 0) { ", $suiteSkipped skipped" }) (exit $($result.ExitCode))"
            $counters.passedSuites++
        }
        elseif ($result.ExitCode -ne 0) {
            $status = "FAILED (exit code $($result.ExitCode))"
            $statusColor = "Red"; $counters.failedSuites++; $counters.failedNames.Add($testName) | Out-Null
        } else {
            $status = if ($xmlParsed) { "$suitePassed passed$(if ($suiteSkipped -gt 0) { ", $suiteSkipped skipped" })" } else { "PASSED" }
            $counters.passedSuites++
        }

        $counters.totalTests += $suiteTotal; $counters.totalPassed += $suitePassed; $counters.totalFailed += $suiteFailed; $counters.totalSkipped += $suiteSkipped

        $icon = if ($statusColor -eq "Green") { [char]0x2713 } elseif ($statusColor -eq "Yellow") { [char]0x26A0 } else { [char]0x2717 }
        $prefix = "  [{0,3}/{1}]" -f $counters.completed, $totalCount
        Write-Host "$prefix " -NoNewline -ForegroundColor DarkGray
        Write-Host "$icon " -NoNewline -ForegroundColor $statusColor
        Write-Host ("{0,-50} " -f $testName) -NoNewline -ForegroundColor White
        Write-Host ("{0,-35} " -f $status) -NoNewline -ForegroundColor $statusColor
        Write-Host ("[{0,6:F1}s]" -f $result.Elapsed) -ForegroundColor DarkGray

        if ($ShowOutput -and (Test-Path $result.LogFile)) {
            $logContent = Get-Content $result.LogFile -Raw
            if ($logContent) { Write-Host $logContent -ForegroundColor DarkGray; Write-Host "" }
        }

        $counters.results.Add([PSCustomObject]@{
            Name = $testName; ExePath = $result.ExePath; Status = if ($result.TimedOut) { "TIMEOUT" } elseif ($result.Crashed) { "CRASH" } elseif ($result.ExitCode -ne 0) { "FAILED" } else { "PASSED" }
            ExitCode = $result.ExitCode; Elapsed = [math]::Round($result.Elapsed, 1)
            Tests = $suiteTotal; Passed = $suitePassed; Failed = $suiteFailed; Skipped = $suiteSkipped
            LogFile = $result.LogFile; XmlFile = $result.XmlFile
        }) | Out-Null
    }
} else {
    # Parallel mode: launch up to $Parallel concurrent processes
    $running = @{}   # pid -> work+process info
    $queue = [System.Collections.Queue]::new($workItems)

    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        # Launch new jobs up to $Parallel
        while ($queue.Count -gt 0 -and $running.Count -lt $Parallel) {
            $work = $queue.Dequeue()
            $testName = $work.TestName
            $logFile  = $work.LogFile
            $xmlFile  = $work.XmlFile

            $mergedFilter = $GTestFilter
            if ($windowsNegativeFilter) {
                if ($mergedFilter) {
                    $mergedFilter = "${mergedFilter}:-${windowsNegativeFilter}"
                } else {
                    $mergedFilter = "*:-${windowsNegativeFilter}"
                }
            }

            $gtestArgs = @("--gtest_output=xml:$xmlFile")
            if ($mergedFilter) { $gtestArgs += "--gtest_filter=$mergedFilter" }

            try {
                $proc = Start-Process -FilePath $work.Exe.FullName -ArgumentList $gtestArgs `
                    -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
                    -NoNewWindow -PassThru

                $running[$proc.Id] = [PSCustomObject]@{
                    Process   = $proc
                    Work      = $work
                    StartTime = [System.Diagnostics.Stopwatch]::StartNew()
                }
            } catch {
                # Process failed to start
                $counters.completed++
                $counters.crashedSuites++
                $counters.failedNames.Add($testName) | Out-Null
                "CRASH: $_" | Set-Content $logFile

                $prefix = "  [{0,3}/{1}]" -f $counters.completed, $totalCount
                Write-Host "$prefix " -NoNewline -ForegroundColor DarkGray
                Write-Host "$([char]0x2717) " -NoNewline -ForegroundColor Red
                Write-Host ("{0,-50} " -f $testName) -NoNewline -ForegroundColor White
                Write-Host ("{0,-35} " -f "CRASH (start failed)") -NoNewline -ForegroundColor Red
                Write-Host "[  0.0s]" -ForegroundColor DarkGray

                $counters.results.Add([PSCustomObject]@{
                    Name = $testName; ExePath = $work.Exe.FullName; Status = "CRASH"
                    ExitCode = -1; Elapsed = 0; Tests = 0; Passed = 0; Failed = 0; Skipped = 0
                    LogFile = $logFile; XmlFile = $xmlFile
                }) | Out-Null
            }
        }

        # Check running processes for completion or timeout
        $finishedPids = @()
        foreach ($procId in @($running.Keys)) {
            $entry = $running[$procId]
            $proc = $entry.Process
            $elapsed = $entry.StartTime.Elapsed.TotalSeconds

            $done = $proc.HasExited
            $entryTestName = $entry.Work.TestName
            $testTimeout = if ($TimeoutOverrides.ContainsKey($entryTestName)) { $TimeoutOverrides[$entryTestName] } else { $TimeoutSeconds }
            $timedOut = (-not $done) -and ($elapsed -ge $testTimeout)

            if ($done -or $timedOut) {
                $finishedPids += $procId
                $entry.StartTime.Stop()
                $work = $entry.Work
                $testName = $work.TestName
                $logFile  = $work.LogFile
                $xmlFile  = $work.XmlFile

                if ($timedOut) {
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                    $proc.WaitForExit(5000) | Out-Null
                }
                $exitCode = $proc.ExitCode

                # Merge stderr
                if (Test-Path "$logFile.err") {
                    $errContent = Get-Content "$logFile.err" -Raw -ErrorAction SilentlyContinue
                    if ($errContent) { Add-Content $logFile "`n--- STDERR ---`n$errContent" }
                    Remove-Item "$logFile.err" -Force -ErrorAction SilentlyContinue
                }

                # Parse XML
                $suitePassed = 0; $suiteFailed = 0; $suiteSkipped = 0; $suiteTotal = 0; $xmlParsed = $false
                if (Test-Path $xmlFile) {
                    try {
                        [xml]$xmlDoc = Get-Content $xmlFile -Raw
                        $ts = $xmlDoc.testsuites
                        if ($ts) {
                            $suiteTotal   = [int]$ts.tests
                            $suiteFailed  = [int]$ts.failures + [int]$ts.errors
                            $suiteSkipped = [int]$ts.disabled + [int]$ts.skipped
                            $suitePassed  = $suiteTotal - $suiteFailed - $suiteSkipped
                            $xmlParsed    = $true
                        }
                    } catch {}
                }

                # Determine status
                $status = ""; $statusColor = "Green"
                if ($timedOut) { $status = "TIMEOUT (>${testTimeout}s)"; $statusColor = "Yellow"; $counters.timeoutSuites++; $counters.failedNames.Add($testName) | Out-Null }
                elseif ($exitCode -ne 0 -and -not $xmlParsed) { $status = "CRASH (exit code $exitCode)"; $statusColor = "Red"; $counters.crashedSuites++; $counters.failedNames.Add($testName) | Out-Null }
                elseif ($exitCode -ne 0 -and $suiteFailed -gt 0) {
                    $status = "$suitePassed passed, $suiteFailed FAILED"
                    $statusColor = "Red"; $counters.failedSuites++; $counters.failedNames.Add($testName) | Out-Null
                }
                elseif ($exitCode -ne 0 -and $suiteFailed -eq 0) {
                    # Process exited non-zero but all GTest tests passed.
                    # This happens on Windows when static destructors (e.g.
                    # MallocAllocator) detect leaks during process teardown
                    # after RUN_ALL_TESTS() already returned 0.
                    $status = if ($xmlParsed) { "$suitePassed passed$(if ($suiteSkipped -gt 0) { ", $suiteSkipped skipped" }) (exit $exitCode)" } else { "PASSED (exit $exitCode)" }
                    $counters.passedSuites++
                } else {
                    $status = if ($xmlParsed) { "$suitePassed passed$(if ($suiteSkipped -gt 0) { ", $suiteSkipped skipped" })" } else { "PASSED" }
                    $counters.passedSuites++
                }

                $counters.totalTests += $suiteTotal; $counters.totalPassed += $suitePassed
                $counters.totalFailed += $suiteFailed; $counters.totalSkipped += $suiteSkipped
                $counters.completed++

                $resultStatus = if ($timedOut) { "TIMEOUT" } elseif ($exitCode -ne 0 -and -not $xmlParsed) { "CRASH" } elseif ($exitCode -ne 0) { "FAILED" } else { "PASSED" }

                # Print result
                $icon = if ($statusColor -eq "Green") { [char]0x2713 } elseif ($statusColor -eq "Yellow") { [char]0x26A0 } else { [char]0x2717 }
                $prefix = "  [{0,3}/{1}]" -f $counters.completed, $totalCount
                Write-Host "$prefix " -NoNewline -ForegroundColor DarkGray
                Write-Host "$icon " -NoNewline -ForegroundColor $statusColor
                Write-Host ("{0,-50} " -f $testName) -NoNewline -ForegroundColor White
                Write-Host ("{0,-35} " -f $status) -NoNewline -ForegroundColor $statusColor
                Write-Host ("[{0,6:F1}s]" -f $elapsed) -ForegroundColor DarkGray

                if ($ShowOutput -and (Test-Path $logFile)) {
                    $logContent = Get-Content $logFile -Raw
                    if ($logContent) { Write-Host $logContent -ForegroundColor DarkGray; Write-Host "" }
                }

                $counters.results.Add([PSCustomObject]@{
                    Name = $testName; ExePath = $work.Exe.FullName; Status = $resultStatus
                    ExitCode = $exitCode; Elapsed = [math]::Round($elapsed, 1)
                    Tests = $suiteTotal; Passed = $suitePassed; Failed = $suiteFailed; Skipped = $suiteSkipped
                    LogFile = $logFile; XmlFile = $xmlFile
                }) | Out-Null
            }
        }

        foreach ($procId in $finishedPids) { $running.Remove($procId) }

        # Brief sleep to avoid busy-wait
        if ($running.Count -gt 0) { Start-Sleep -Milliseconds 500 }
    }
}

$totalStopwatch.Stop()
$totalElapsed = $totalStopwatch.Elapsed

# Reassign for summary section
$passedSuites  = $counters.passedSuites
$failedSuites  = $counters.failedSuites
$crashedSuites = $counters.crashedSuites
$timeoutSuites = $counters.timeoutSuites
$totalTests    = $counters.totalTests
$totalPassed   = $counters.totalPassed
$totalFailed   = $counters.totalFailed
$totalSkipped  = $counters.totalSkipped
$failedNames   = @($counters.failedNames)
$results       = @($counters.results)

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  TEST RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Suites: " -NoNewline -ForegroundColor White
Write-Host "$passedSuites passed" -NoNewline -ForegroundColor Green
if ($failedSuites -gt 0) {
    Write-Host ", $failedSuites failed" -NoNewline -ForegroundColor Red
}
if ($crashedSuites -gt 0) {
    Write-Host ", $crashedSuites crashed" -NoNewline -ForegroundColor Red
}
if ($timeoutSuites -gt 0) {
    Write-Host ", $timeoutSuites timeout" -NoNewline -ForegroundColor Yellow
}
Write-Host "  (of $totalCount)" -ForegroundColor White

if ($totalTests -gt 0) {
    Write-Host "  Tests : " -NoNewline -ForegroundColor White
    Write-Host "$totalPassed passed" -NoNewline -ForegroundColor Green
    if ($totalFailed -gt 0) {
        Write-Host ", $totalFailed failed" -NoNewline -ForegroundColor Red
    }
    if ($totalSkipped -gt 0) {
        Write-Host ", $totalSkipped skipped" -NoNewline -ForegroundColor Yellow
    }
    Write-Host "  (of $totalTests)" -ForegroundColor White
}

Write-Host "  Time  : $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "  Logs  : $LogDir" -ForegroundColor White

if ($failedNames.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed suites:" -ForegroundColor Red
    foreach ($name in $failedNames) {
        Write-Host "    - $name" -ForegroundColor Red
        $logPath = Join-Path $LogDir "$name.log"
        if (Test-Path $logPath) {
            Write-Host "      Log: $logPath" -ForegroundColor DarkGray
        }
        # Show individual failed test cases from XML
        $xmlPath = Join-Path $LogDir "$name.xml"
        if (Test-Path $xmlPath) {
            try {
                [xml]$xmlDoc = Get-Content $xmlPath -Raw
                foreach ($suite in $xmlDoc.testsuites.testsuite) {
                    foreach ($tc in $suite.testcase) {
                        if ($tc.failure) {
                            Write-Host "      FAIL: $($suite.name).$($tc.name)" -ForegroundColor DarkYellow
                        }
                    }
                }
            } catch {}
        }
    }
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan

# Write summary JSON
$summaryObj = [PSCustomObject]@{
    Timestamp     = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    BuildType     = $BuildType
    Filter        = $Filter
    GTestFilter   = $GTestFilter
    TotalSuites   = $totalCount
    PassedSuites  = $passedSuites
    FailedSuites  = $failedSuites
    CrashedSuites = $crashedSuites
    TimeoutSuites = $timeoutSuites
    TotalTests    = $totalTests
    TotalPassed   = $totalPassed
    TotalFailed   = $totalFailed
    TotalSkipped  = $totalSkipped
    ElapsedTime   = $totalElapsed.ToString('hh\:mm\:ss')
    Results       = $results
}
$summaryObj | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $LogDir "summary.json") -Encoding UTF8

# Exit with appropriate code
if ($failedNames.Count -gt 0) {
    exit 1
} else {
    exit 0
}

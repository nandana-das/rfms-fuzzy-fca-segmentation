$csvPath = "results/fuzzy_tnorm_tail_ablation/multi_origin_metrics.csv"
$rows = Import-Csv $csvPath

function Compute-Stats($deltas) {
    $n = $deltas.Count
    $pos = ($deltas | Where-Object { $_ -gt 0 }).Count
    $zero = ($deltas | Where-Object { $_ -eq 0 }).Count
    $neg = ($deltas | Where-Object { $_ -lt 0 }).Count
    
    $sum = 0.0
    foreach ($d in $deltas) { $sum += $d }
    $mean = $sum / $n
    
    $sorted = $deltas | Sort-Object
    if ($n % 2 -eq 1) {
        $median = $sorted[[Math]::Floor($n / 2)]
    } else {
        $mid = $n / 2
        $median = ($sorted[$mid - 1] + $sorted[$mid]) / 2.0
    }
    
    $min = $sorted[0]
    $max = $sorted[-1]
    $consistent = ($pos -eq $n) -or ($neg -eq $n)
    
    return [PSCustomObject]@{
        Pos = $pos
        Zero = $zero
        Neg = $neg
        Mean = $mean
        Median = $median
        Min = $min
        Max = $max
        Consistent = $consistent
    }
}

$datasets = @("Dunnhumby", "Online Retail II")
$metrics = @("auc", "spend_r2", "invoice_r2")
$variants = @("M1", "M2", "M3")

Write-Output "================================================================="
Write-Output "DESCRIPTIVE COMPARISON OF FACTORIAL ARMS (M1, M2, M3 vs M0)"
Write-Output "================================================================="

foreach ($ds in $datasets) {
    Write-Output "`n#################################################################"
    Write-Output "DATASET: $ds"
    Write-Output "#################################################################"
    $dsRows = $rows | Where-Object { $_.dataset -eq $ds }
    $origins = $dsRows | Select-Object -ExpandProperty origin -Unique
    
    foreach ($m in $metrics) {
        Write-Output "`n-----------------------------------------------------------------"
        Write-Output "Metric: $m"
        Write-Output "-----------------------------------------------------------------"
        
        # Collect M0 values
        $m0Vals = @{}
        foreach ($orig in $origins) {
            $r0 = $dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq "M0" }
            $m0Vals[$orig] = [double]$r0.$m
        }
        
        foreach ($v in $variants) {
            Write-Output "`nVariant: $v vs M0 ($m)"
            $deltas = @()
            foreach ($orig in $origins) {
                $rv = $dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq $v }
                $vVal = [double]$rv.$m
                $m0Val = $m0Vals[$orig]
                $delta = $vVal - $m0Val
                $deltas += $delta
                Write-Output ("  Origin {0,-12}: M0 = {1:F8}, {2} = {3:F8}, Delta = {4:+0.00000000;-0.00000000; 0.00000000} (raw: {5:R})" -f $orig, $m0Val, $v, $vVal, $delta, $delta)
            }
            $stats = Compute-Stats $deltas
            Write-Output ("  Summary: Pos={0}, Zero={1}, Neg={2} | Mean={3:+0.00000000;-0.00000000; 0.00000000} | Median={4:+0.00000000;-0.00000000; 0.00000000} | Min={5:+0.00000000;-0.00000000; 0.00000000} | Max={6:+0.00000000;-0.00000000; 0.00000000} | Consistent={7}" -f $stats.Pos, $stats.Zero, $stats.Neg, $stats.Mean, $stats.Median, $stats.Min, $stats.Max, $stats.Consistent)
        }
    }
}

Write-Output "`n================================================================="
Write-Output "FACTORIAL DESIGN ANALYSIS (MAIN EFFECTS AND INTERACTION)"
Write-Output "================================================================="

foreach ($ds in $datasets) {
    Write-Output "`n#################################################################"
    Write-Output "DATASET: $ds - FACTORIAL CONTRASTS"
    Write-Output "#################################################################"
    $dsRows = $rows | Where-Object { $_.dataset -eq $ds }
    $origins = $dsRows | Select-Object -ExpandProperty origin -Unique
    
    foreach ($m in $metrics) {
        Write-Output "`nMetric: $m"
        
        $m1_m0_list = @()
        $m3_m2_list = @()
        $m2_m0_list = @()
        $m3_m1_list = @()
        $interaction_list = @()
        
        foreach ($orig in $origins) {
            $m0 = [double]($dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq "M0" }).$m
            $m1 = [double]($dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq "M1" }).$m
            $m2 = [double]($dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq "M2" }).$m
            $m3 = [double]($dsRows | Where-Object { $_.origin -eq $orig -and $_.arm -eq "M3" }).$m
            
            $f1_at_min = $m1 - $m0
            $f1_at_prod = $m3 - $m2
            $f2_at_bl = $m2 - $m0
            $f2_at_lar = $m3 - $m1
            $inter = ($m3 - $m1) - ($m2 - $m0) # = (M3 - M2) - (M1 - M0)
            
            $m1_m0_list += $f1_at_min
            $m3_m2_list += $f1_at_prod
            $m2_m0_list += $f2_at_bl
            $m3_m1_list += $f2_at_lar
            $interaction_list += $inter
            
            Write-Output ("  Origin {0,-12}: F1(Min)={1:+0.000000;-0.000000}, F1(Prod)={2:+0.000000;-0.000000} | F2(BL)={3:+0.000000;-0.000000}, F2(LAR)={4:+0.000000;-0.000000} | Inter={5:+0.000000;-0.000000}" -f $orig, $f1_at_min, $f1_at_prod, $f2_at_bl, $f2_at_lar, $inter)
        }
        $s_f1_min = Compute-Stats $m1_m0_list
        $s_f1_prod = Compute-Stats $m3_m2_list
        $s_f2_bl = Compute-Stats $m2_m0_list
        $s_f2_lar = Compute-Stats $m3_m1_list
        $s_inter = Compute-Stats $interaction_list
        
        Write-Output ("  Mean Contrasts: F1|Min={0:+0.00000000} | F1|Prod={1:+0.00000000} | F2|BL={2:+0.00000000} | F2|LAR={3:+0.00000000} | Inter={4:+0.00000000}" -f $s_f1_min.Mean, $s_f1_prod.Mean, $s_f2_bl.Mean, $s_f2_lar.Mean, $s_inter.Mean)
        Write-Output ("  Median Contrasts: F1|Min={0:+0.00000000} | F1|Prod={1:+0.00000000} | F2|BL={2:+0.00000000} | F2|LAR={3:+0.00000000} | Inter={4:+0.00000000}" -f $s_f1_min.Median, $s_f1_prod.Median, $s_f2_bl.Median, $s_f2_lar.Median, $s_inter.Median)
    }
}

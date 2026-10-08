# Canonical Kuznetsov Intensional Stability - Audit Report

> **Note (2026-10-08 audit).** The exact stability computation documented here was verified correct. The Kuznetsov filter is nevertheless **excluded from the final method**: it gives no consistent predictive benefit, it is the least stable arm, and its threshold acts as an absolute customer-count rule that depends on training-set size. These files are kept as ablation evidence; see `docs/AUDIT_ERRATA.md` §7.

Generated: 2026-10-06 16:32:40 | script: `scripts/kuznetsov_stability_prototype.py`
| branch: experiment/canonical-kneedle-full | total wall time: 51.1s

Isolated prototype. No production pipeline, script, result, or methodology
document was modified; nothing was committed; no pruning, Kneedle, Jaccard,
or predictive machinery was touched.

## 1. Executive summary

- Exact canonical intensional stability Stab(A,B) = |{X subseteq A: X' = B}| / 2^|A|
  (Kuznetsov 2007) was computed for **861 of 861 reconstructed
  concepts (0 failures)** across Dunnhumby and Retail II.
- Every concept passed an A' = B re-derivation check (861/861);
  861/861 reconstructed extents matched the production `n_customers` counts.
- Verification: hand-computed toys PASS, randomized
  lattice + brute-force validation PASS,
  Monte-Carlo cross-checks on 18 real concepts, brute-force guard
  triggered on 0 concept(s) (extents <= 20: none expected, none found).
- Verdict: **feasible (A: YES); the production proxy is NOT canonical
  stability and rank-agrees only weakly (pooled Spearman 0.3343).**

## 2. Contexts reconstructed (unchanged production functions)

| dataset | objects | attributes | concepts | A'=B verified | exact | failed |
| --- | --- | --- | --- | --- | --- | --- |
| Dunnhumby Observation (Days 1-620) | 2,499 | 45 | 502 | 502/502 | 502/502 | 0 |
| UK Online Retail II (Full Cohort) | 5,878 | 45 | 359 | 359/359 | 359/359 | 0 |

Retail II reference cross-check vs `controlled_comparison_concepts_fuzzy.csv`: available: 359/359 intents matched, 0 customer-count mismatches.

## 3. Verification evidence

- Toy contexts: 4 contexts, 13 concepts, 0 mismatches (all hand-computed).
- Randomized small contexts: 40 contexts, 231 concepts;
  0 children mismatches (attribute-based vs independently
  enumerated lattice), 0 generator-count mismatches
  (inclusion-exclusion vs exhaustive submask enumeration).
- Brute-force guard (extents <= 20): 0 concept(s) checked (real extents are all larger).

### Monte-Carlo cross-check (1,000,000 uniform subset samples per concept)

| dataset | concept | extent | exact | MC estimate | observed/expected generators | 4-sigma |
| --- | --- | --- | --- | --- | --- | --- |
| Dunnhumby Observatio | DH_C_491 | 103 | 4.9609e-01 | 4.9598e-01 | 495,976 / 496094 | OK |
| Dunnhumby Observatio | DH_C_451 | 113 | 8.7500e-01 | 8.7522e-01 | 875,218 / 875000 | OK |
| Dunnhumby Observatio | DH_C_308 | 169 | 9.8437e-01 | 9.8465e-01 | 984,652 / 984375 | OK |
| Dunnhumby Observatio | DH_C_346 | 154 | 9.9902e-01 | 9.9904e-01 | 999,038 / 999023 | OK |
| Dunnhumby Observatio | DH_C_275 | 181 | 1.0000e+00 | 1.0000e+00 | 999,997 / 999996 | OK |
| Dunnhumby Observatio | DH_C_041 | 544 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| Dunnhumby Observatio | DH_C_187 | 246 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| Dunnhumby Observatio | DH_C_003 | 1,922 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| Dunnhumby Observatio | DH_C_031 | 633 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_272 | 321 | 5.0000e-01 | 4.9962e-01 | 499,619 / 500000 | OK |
| UK Online Retail II  | R2_C_071 | 880 | 9.6875e-01 | 9.6874e-01 | 968,742 / 968750 | OK |
| UK Online Retail II  | R2_C_249 | 345 | 9.9997e-01 | 9.9996e-01 | 999,963 / 999969 | OK |
| UK Online Retail II  | R2_C_145 | 562 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_186 | 461 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_126 | 627 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_107 | 712 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_015 | 1,951 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |
| UK Online Retail II  | R2_C_003 | 4,904 | 1.0000e+00 | 1.0000e+00 | 1,000,000 / 1000000 | OK |

## 4. Canonical stability distributions (log2 scale; float64 collapse at 1.0 avoided)

Log2 range is the natural reporting scale here: raw float64 stability values
group to exactly 1.0 for every concept within 2^-53 of 1, obscuring the
exact separation. All order statistics are computed from the exact
Decimal-derived log2 (preserves losses down to ~1e-18).

Symbols: n = concepts; log2 values are log2(stability) so 0 = exactly 1,
-1 = 1/2, -113 = within 2^-113 of 1; stab_min/max are 2^log2 values;
distinct = exact rationals distinct vs float64-distinct; n_exact_one counts
concepts whose exact rational equals 1; loss<2^-K columns count concepts whose
loss (1-stability) is below 2^-K i.e. stability is within 2^-K of 1 (essentially perfect).
stab<2^-K columns (all 0 here) would count astronomically unstable concepts.

| dataset | n | log2 range [min, p05, p25, med, p75, p95, max] | stab range [min, max] | log2 med/max | distinct (exact/float) | ==1 | loss<2^-40 | loss<2^-80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dunnhumby Observation (Days 1-620) | 502 | log2[ -0.910, -0.902, -0.046, -0.000, -0.000, -0.000, -0.000 ] | stab[ 0.532, 1.000 ] | -0.000/-0.000 | 500/227 | 0 | 113 | 41 |
| UK Online Retail II (Full Cohort) | 359 | log2[ -0.902, -0.902, -0.000, -0.000, -0.000, -0.000, 0.000 ] | stab[ 0.535, 1.000 ] | -0.000/0.000 | 359/82 | 0 | 167 | 103 |

### Proxy-vs-canonical summary

| grouping | n | Pearson r | Spearman rho | Kendall tau | top-10 / top-20 / top-50 overlap (overlap/k) |
| --- | --- | --- | --- | --- | --- |
| POOLED | 861 | 0.0284 | 0.3343 | 0.2246 | 10: 0/10 / 20: 2/20 / 50: 8/50 |
| Dunnhumby Observation (Days 1-620) | 502 | -0.0096 | 0.2443 | 0.1611 | 10: 0/10 / 20: 2/20 / 50: 9/50 |
| UK Online Retail II (Full Cohort) | 359 | 0.0462 | 0.1788 | 0.1233 | 10: 0/10 / 20: 2/20 / 50: 11/50 |

### Anchor ranges per dataset (proxy min/med/max vs canonical value range)

| dataset | proxy [min, med, max] (distinct) | canonical [min, median, max] (distinct exact/float) |
| --- | --- | --- |
| Dunnhumby Observation (Days 1-620) | proxy [0.804, 0.977, 0.995] (424 distinct) | canonical [0.532, 1.000, 1.000] (500/227 exact/float-distinct) |
| UK Online Retail II (Full Cohort) | proxy [0.957, 0.991, 0.998] (345 distinct) | canonical [0.535, 1.000, 1.000] (359/82 exact/float-distinct) |

### Most and least stable concepts (by exact log2)

| dataset | rank | concept | intent (truncated) | extent | exact stability (Fraction) | log2 | proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dunnhumby Observation (Days 1-620) | top-5 | DH_C_003 | R5@0.3 & R5@0.5 & R5@0.7 | 1922 | 379884580723156565621879454783539879962769882538740466971203825322340665146784135500423149633284420064691795393551919597524315106192234877096612153927677320307949850353728791113387899582551021379426437331597087128309844032728477446773103143657831643868597613891244356268381828074624508255164810237562162920075274526381374484702185619551039595871989083239473375500595641127969670066409521923775623364688707626065041544109776835882255496579354015655651316126017271193087149668672135159223311705338023559217875405184190591102101790365708125294282109095493425602320466492288285103445/379884580723156565621879454783539879962769882538740466971203825322340665146784135500423149633284420064691795393551919597524315106192234877096612153927677320307949850353728791113387899582551021379426437331597087128309844032728477446773103143657831643868597613891244356268381828074624508255164810237564431927809161989654686064756218860120848605830919702986429364272429802224146765013782080632701771065092427217842322395041703183725628235285452051918629148053059185252089068563632917965139439989701359880029576238563434145991515663112656098818055313002363591794418357125233774690304 | -0.0 | 0.9891 |
| Dunnhumby Observation (Days 1-620) | top-5 | DH_C_031 | F1@0.3 & F1@0.5 & F1@0.7 & R5@0.3 & R5@0.5 & R5@0.7 | 633 | 278469275977917188637766821636980671685377716506870112950390168328813181592326500641230029987098534965788478588194224368720719897221762920646577458048074831780233404895703632049447907885055/278469275977917188637766821636980671685377716506870112950390168328813181592326500641230029987632531724770648777685205524334027387779262573135538410627576549575927409323746640019706676772864 | -0.0 | 0.9921 |
| Dunnhumby Observation (Days 1-620) | top-5 | DH_C_007 | F1@0.3 & F1@0.5 & F1@0.7 | 1084 | 207259907386686073192955235040171322418746009358481529197466990620088667717627234998869935973664550630022984254706239900423723441435373920447245745655090759020357663802035641296547849012731791489610502179940697543735240432481876276030751484869597630290184415569454256774653246811213840239450362800891112580004083506999590422887/207259907386686073192955235040171322418746009358481529197466990620088667717627235217367432910271615483071567733060415397263164147083052520311821721539316234159633842459203954037214113527903453533429522102756090950207495663026885323384937836184739385692989497868655380021589416565996034592958799368853510413627092726879738658816 | -0.0 | 0.9825 |
| Dunnhumby Observation (Days 1-620) | top-5 | DH_C_015 | F2@0.3 & R5@0.3 & R5@0.5 & R5@0.7 | 889 | 515912628062173092140956821207535748553561841832149923953086629908861232947301858109996820578225536130591798561029750696476802816260059916519061683462840048961539275471527786744469227937813658052155609381493567025936325707363867473182400324796785734408508074558018357/515912628062173092140956821207535748553561841832149923953086629908861232965551620580485601452790222553392963860602664725471042538576841344971229471155214430574596371092778402508526872730885196340843977449424988675925808879640197010584371971452184059071815020407947264 | -0.0 | 0.9843 |
| Dunnhumby Observation (Days 1-620) | top-5 | DH_C_030 | F2@0.3 & F2@0.5 & R5@0.3 & R5@0.5 & R5@0.7 | 645 | 18249762470488780874564686422801165299572914028994239722316770071597100627518711496633113806295289362710770827889023282117776575945049423876936662637640877895408892400212452911282833807393285387/18249762470488780874564686422801165299572914028994239722316770071597100668834709546023651245269485599114569238294377629242754818885501751993010645278888856753007978697441059800331496768986415104 | -0.0 | 0.9845 |
| Dunnhumby Observation (Days 1-620) | bottom-5 | DH_C_491 | F4@0.3 & F4@0.5 & M5@0.3 & M5@0.5 & R5@0.3 | 103 | 8727373545345/17592186044416 | -0.9 | 0.9709 |
| Dunnhumby Observation (Days 1-620) | bottom-5 | DH_C_387 | F4@0.3 & M5@0.3 & M5@0.5 & R5@0.3 | 136 | 1099243192193/2199023255552 | -0.9 | 0.9779 |
| Dunnhumby Observation (Days 1-620) | bottom-5 | DH_C_391 | F4@0.3 & M5@0.3 & M5@0.5 & R5@0.3 & R5@0.5 | 135 | 633670557609204028213989212129/1267650600228229401496703205376 | -0.9 | 0.9778 |
| Dunnhumby Observation (Days 1-620) | bottom-5 | DH_C_500 | M1@0.3 & R4@0.3 & R4@0.5 | 100 | 18012199486226177/36028797018963968 | -0.9 | 0.9500 |
| Dunnhumby Observation (Days 1-620) | bottom-5 | DH_C_378 | F4@0.3 & F4@0.5 | 139 | 2658415426750624296045735140988027395/5316911983139663491615228241121378304 | -0.9 | 0.9424 |
| UK Online Retail II (Full Cohort) | top-5 | R2_C_003 | F1@0.3 & F1@0.5 & F1@0.7 | 4904 | 222848004313668719691213483700810759987656019547710139664241936991138106539616439762775058065688367531907719095700069508131964048744509937894796740738505066831483388299980505865722101281973991752051079878406943136495731052685385981167936406108052259153180314590401441086107951134763582222178806130979450523610292690241037876575460384059273115066548829780705991363970146861803501538571064868730128783979777341880332463641995130956398538891263597839918422762242505609422357073557925121726514010836900537641686656503152430580165941404831443945570282690367110620682083028935478555282639476176629026776268383873097027471871476729615653553852994769678614918913374970497476655715836999681683299634189187555967856521001252998565559074621638218445299839894876830414258864588821502084613261987198856784541783369197220600791309913091912501503201230610555510659802846749449219149464300609873734604409036282729377993730736333607926346613405190282127985534677358727614469681715362087496658867153320675585185570748539234688362492109138752512035022435745244585590313987789985927697194348774318163719610137786499641071042145508795606602284236865789643227288718891718277301347048898522956931058177599017017953629577293346805981566806486498071336660570831992809125098017235397954930372303841395117086869881387720681754187740845933552392260533897561538901920113093247084963739938409521051304127486315357655382735139592151441908775491478482922746742626351937735530503127213945814579156818660622287/222848004313668719691213483700810759987656019547710139664241936991138106539616439762775058065688367531907719095700069508131964048744509937894796740738505066831483388299980505865722101281973991752051079878406943136495731052685385981167936406108052259153180314590401441086107951134763582222178806130979450523610292690241037876575460384059273115066548829780705991363970146861803501538571064868730128783979777341880332463641995130956398538891263597839918422762242505609422357073557925121726514010836900537641686656503152430580165941404831443945570282690367110620682083028935478555282639476176629026776268383873097027471871476729615653553852994769678614918913374970497476655715836999681683299634189187555967856521001252998565559074621638218445299839894876830416575083610618206257432118972401116133269463833866230950916952995411373918148349361215693246613829888213288305357239073545221307206167441574530195787313912090912979929382777969487951949322618906724196994628638037640106093305472621641881937827447206916332715678892320927194478168405438778896510491602870644308869938666086561401068093659871427096523225215861650585301063106894861911088059286678346350307257457833611661977586521721847940184057530922580500894412994920507883101337204349029037389584516440678117026220089119744731571552913544501272187787585483807869056891982387618102642674500927575453725610336484859306845245953350338792277922673893920973157108631164802759853751459885883070656697593169142468287187573383626752 | -0.0 | 0.9949 |
| UK Online Retail II (Full Cohort) | top-5 | R2_C_015 | F1@0.3 & F1@0.5 & F1@0.7 & R5@0.3 & R5@0.5 & R5@0.7 | 1951 | 9819786255688460528742238970227602814817438663889495986280639962151666144621084648439944216495781327947644161398532399291553744506412482488781663545817807448211710863491101520258503344506201776990026443246709472122781084691000091345909940069444727130384881225551306268868388835644097433646376281013558439642207351183769808340943425117215006718884100350841550025909675785765118035779988807518131674674370254940368683019147132073894792950646373905399516987017869889546230361592026627615428222532430996014564473247219956434949345275360240298218778508918785/9819786255688460528742238970227602814817438663889495986280639962151666144621084648439944216495781327947644161398532399291553744506412482488781663545817807448211710863491101520258503344506201776990026443246709472122781084691000091345909940069444727130384881225551306268868388835644097433646376281013558439642207351183769808340943425117215006718884100350841550062409479196018657702097999419942099254491883690645354997765637676527103720284917230766999974128311514396157779102391271000567722480231697745772085750410670046837510834828141407629302350030569472 | -0.0 | 0.9974 |
| UK Online Retail II (Full Cohort) | top-5 | R2_C_054 | F1@0.3 & F1@0.5 & F1@0.7 & M1@0.3 & M1@0.5 & M1@0.7 | 1005 | 19033816428515623203815199976318727169680130581240249075913879799244040411653175981378154425550801287549423664514470055045818691142974793059722631438110651210022026757727486386466386045879011031939061706014098396237665449585644081724222482582271047542924841515396193615520212056828542977/19033816428515623203815199976318727169680130581240249075913879799244040411653175981378154425550801287549423664514470055045818691142974793059722631438110651210022026757727486386466386045879011031939061706014098396237667183448036865128410866436462823462554177349813042084144196464827957248 | -0.0 | 0.9950 |
| UK Online Retail II (Full Cohort) | top-5 | R2_C_032 | F1@0.3 & F1@0.5 & F1@0.7 & M2@0.3 & M2@0.5 | 1279 | 20327992567703904456885235014382979567607934076692782822888891309382046991549531832055155159633445673520354714754455093342546027828756767398710755089213420082705421653717374176620219823398076816641429074719010471770462584994262724205453330752760051767312320858833997083430572653680452980987829533912347911286534897068161944858351380368315686994374743470786254427116871361677592492343/20327992567703904456885235014382979567607934076692782822888891309382046991549531832055155159633445673520354714754455093342546027828756767398710755089213517115086190448025820987838248729807780372017498868257206662522839213178805622787036135516480977132289743916623360993239094892633311727863720374786950784857555121490508830130270381813931558650152443104953490040458438887804107751424 | -0.0 | 0.9891 |
| UK Online Retail II (Full Cohort) | top-5 | R2_C_096 | F1@0.3 & F1@0.5 & F1@0.7 & M2@0.3 & M2@0.5 & M2@0.7 | 773 | 1445895146858607358437943727208769466035893202868007692637901788601699241144933631951807447549557758449099707135121406247999127995329736103019390630081986137069786580290384715367658370733342864820407779193762644133139185665/1445895146858607358437943727208769466035893202868007692637901788601699241144933631951807447549557758449099707135121406247999127995329736165184795181305316406492567598839150653733187621116264206194563768053163279547256274944 | -0.0 | 0.9935 |
| UK Online Retail II (Full Cohort) | bottom-5 | R2_C_272 | F1@0.3 & M2@0.3 & M2@0.5 & M2@0.7 & R5@0.3 | 321 | 134217727/268435456 | -0.9 | 0.9938 |
| UK Online Retail II (Full Cohort) | bottom-5 | R2_C_301 | F1@0.3 & M2@0.3 & M2@0.5 & M2@0.7 & R5@0.3 & R5@0.5 | 294 | 137438953471/274877906944 | -0.9 | 0.9966 |
| UK Online Retail II (Full Cohort) | bottom-5 | R2_C_322 | F1@0.3 & M1@0.3 & M1@0.5 & R4@0.3 | 267 | 2535301200454152959983118974977/5070602400912917605986812821504 | -0.9 | 0.9888 |
| UK Online Retail II (Full Cohort) | bottom-5 | R2_C_163 | F1@0.3 & M2@0.3 & M2@0.5 & R5@0.3 | 513 | 842498333348265931640735985362068665172766974556440547644080652289/1684996666696914987166688442938726917102321526408785780068975640576 | -0.9 | 0.9883 |
| UK Online Retail II (Full Cohort) | bottom-5 | R2_C_016 | F1@0.3 & M2@0.3 | 1668 | 489378619942776555776259149753645814473086509947526392401563180567357582368735666574294911505263774338526756054988071596096625916485821630712924703870407377045507225423539406898859252550651292008453257057930209053465966980733321023175781971460378125352003013615360590382428818600637205559951590534420024423851112645610506219181575688724234545135617/978757239885553111552518299507291628946173019895052784803126361134715164737471333148589823010527548677053512109976144744711344133680578410405337870243370010977031567543689952849756531152255270393792844994269246753409884449197339177424770114500800365518397471517996221945996841656250431969808731337118835799400739730701352692234849894550672754343936 | -0.9 | 0.9910 |

## 5. Runtime and memory benchmark

| dataset | concepts | exact | failed | reconstruction s | stability s | median ms | max ms | peak tracemalloc/rss MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dunnhumby Observation (Days 1-620) | 502 | 502 | 0 | 7.6 | 2.5 | 1.0 | 13.3 | 2 / nan |
| UK Online Retail II (Full Cohort) | 359 | 359 | 0 | 7.2 | 12.5 | 1.1 | 9.2 | 3 / nan |

Verification stages (toys + randomized validation + Monte-Carlo) took 20.1s.

## 6. Verdict A-I

| letter | topic | one-line outcome |
| --- | --- | --- |
| A | Feasibility | YES |
| B | Algorithm | see below |
| C | Exactness coverage | see below |
| D | Runtime | see below |
| E | Memory | see below |
| F | Proxy vs canonical fidelity | see below |
| G | Discriminativeness | see below |
| H | Pruning suitability | see below |
| I | Recommended next experiment | see below |

### Verdict A - Feasibility

**YES - feasibility demonstrated.** All 861 concepts across 2 datasets (Dunnhumby Observation (Days 1-620), UK Online Retail II (Full Cohort)) were processed and 861/861 returned exact rational values. No concept exceeded the memo-state or wall-clock caps, and no value was reported as a bound. Canonical Kuznetsov intensional stability is computable on this hardware with the production contexts as-is.

### Verdict B - Algorithm

Exact inclusion-exclusion over the maximal lower-neighbor extents (`count_generators_ie`), derived from the standard direct-descendant characterization: non-generators are exactly the subsets of A covered by A & extent(m) for some m not in B, and only the maximal distinct masks matter. The algorithm never enumerates the 2^|A| subsets (extents here reach 4,904 objects in the MC sample; real extents up to thousands). Observed worst case in this run: 12 lower neighbors and 550 memo states for a single concept - trivial for exact integer arithmetic. Counts are Python integers; stability is emitted as an exact Fraction plus a 60-digit decimal and log2/log values.

### Verdict C - Exactness coverage

**861/861 exact (100.0% of reconstructed concepts), 0 failures.** Every concept passed the A' = B re-derivation check before stability was computed, and every reconstructed concept extent matched the table's `n_customers` count (861/861 matches; 0 mismatches recorded in the CSV). Independent verification: toy contexts PASS (13 hand-computed concepts), randomized lattice/brute-force validation PASS (40 contexts, 231 concepts), Retail II reference cross-check (available: 359/359 intents matched), and Monte-Carlo cross-checks (18 concepts, 18/18 within 4 sigma).

### Verdict D - Runtime

Total wall time for the whole prototype: 51.1s (including data loading, mining, verification and reporting). Stability computation proper: 15.0s for 861 concepts. Dunnhumby Observation (Days 1-620): 502/502 exact, 2.5s, median 1.0ms/concept, stability in [2^-0.91, 1] (median loss 2^-0.00); UK Online Retail II (Full Cohort): 359/359 exact, 12.5s, median 1.1ms/concept, stability in [2^-0.90, 1] (median loss 2^-0.00). Per-concept cost is milliseconds; the dominant cost of the prototype is the unchanged production reconstruction (loading + scoring + mining).

### Verdict E - Memory

Peak memory is reported per dataset in `runtime_benchmark.csv` (tracemalloc peak + process peak RSS). The exact-stability stage allocates only Python integers and small numpy arrays per concept; the memoization dictionaries stay tiny (see B), so memory is dominated by the unchanged production data frames and mining.

### Verdict F - Proxy vs canonical fidelity

Pooled Pearson r = 0.0284, Spearman rho = 0.3343, Kendall tau = 0.2246. Rank agreement is positive but weak, and top-k agreement is what matters for pruning: Dunnhumby Observation (Days 1-620): top-20 overlap 2/20, top-50 overlap 9/50; UK Online Retail II (Full Cohort): top-20 overlap 2/20, top-50 overlap 11/50. The proxy saturates in a narrow band near 1 (Dunnhumby Observation (Days 1-620): [0.804, 0.995]), (UK Online Retail II (Full Cohort): [0.957, 0.998]) while canonical stability on these dense fuzzy contexts is even more saturated in value space (Dunnhumby Observation (Days 1-620): min 0.532, median within 2^-1.3e-04 of 1), (UK Online Retail II (Full Cohort): min 0.535, median within 2^-1.0e-11 of 1). Exact rationals still separate concepts that float64 collapses to the same value, but the two measures are plainly not interchangeable as ranking signals (weak rank agreement above).

### Verdict G - Discriminativeness

Canonical stability takes 500 distinct exact rational values (227 after float64 rounding) for Dunnhumby Observation (Days 1-620), 359 distinct exact rational values (82 after float64 rounding) for UK Online Retail II (Full Cohort). The value range is narrow: Dunnhumby Observation (Days 1-620): [0.532, 1] with median loss (1-stability) about 2^-1.3e-04, UK Online Retail II (Full Cohort): [0.535, 1] with median loss (1-stability) about 2^-1.0e-11. Measured as loss = 1 - stability, concepts spread from Dunnhumby Observation (Days 1-620): max loss 0.468 (stability 0.532, the least stable concept) down through 113/502 concepts with loss < 2^-40 (essentially perfect, stability within ~9e-13 of 1) and 41/502 with loss < 2^-80 (stability within ~8e-25 of 1); UK Online Retail II (Full Cohort): max loss 0.465 down through 167/359 with loss < 2^-40 and 103/359 with loss < 2^-80. So the tail of LOWER stability (loss ~0.5, the 76 concepts with stability <= 0.5) coexists with a large core of essentially-perfect concepts: canonical stability IS discriminative via the tail (loss up to 0.468) while the median concept remains extremely close to 1. The proxy's larger float spread does not reflect comparable canonical differences.

### Verdict H - Pruning suitability

As an audit metric, canonical stability is exact, reproducible, and cheap (milliseconds per concept). As a PRUNING axis in this regime it is weak on its own: since nearly every concept is within a vanishing loss of 1, any absolute threshold either prunes almost nothing or must be placed in log-loss space near the extreme tail (loss ~ 0.5, populated by small-extent concepts). What this audit does NOT do (by explicit constraint) is run any pruning, so the practical effect of any cutoff remains an empirical question for a separate experiment.

### Verdict I - Recommended next experiment

Two follow-ups, both isolated prototypes on the same 502 + 359 concepts and untouched production code: (1) a pruning-sensitivity experiment that thresholds on exact log2 loss (and support x loss combinations) and compares surviving sets against the locked Kneedle/proxy outputs; and (2) given the saturation found here, an evaluation of alternative concept-interestingness measures from Kuznetsov & Makhalova (e.g. extensional stability, lift, conviction) to find a signal that actually discriminates in this dense-fuzzy regime. Only after such comparisons should any pipeline change be considered.


## 7. Limitations and compliance

- The reconstruction re-runs the production miners unchanged; all extents were
  re-verified via A' = B, so no drift between reconstruction and production is
  possible without detection.
- Resource caps (500,000 memo states, 120s per concept)
  bound only a worst case that did not occur; a hit would be reported as
  `failed:*` and excluded from exact counts, never replaced by an approximation.
- Monte-Carlo checks are cross-checks only; every reported stability value is
  an exact rational, never a bound or an estimate.
- No pruning, no Kneedle, no Jaccard suppression change, no predictive
  experiment; no production file modified; nothing committed.

## 8. Artifact inventory

- `stability_values.csv` - one row per concept: exact generator count, exact
  Fraction, 60-digit decimal, log2 stability, audit flags, timings.
- `runtime_benchmark.csv` - per-dataset timings, memory peaks, failure counts.
- `algorithm_description.md` - definitions, theorem, algorithm, complexity,
  verification plan, citations.
- this report.

# HN-12 — Design-based inference (R survey package) for the co-author revision.
# Taylor-series linearisation with masked strata (SDMVSTRA) and PSUs (SDMVPSU); the analytic cohort and every group
# are analysed as subpopulations of the full examined sample; CIs and P values use design degrees of freedom.
# Input: hn11_analysis.csv.gz from HN-11. Output: hn12_results.json, km_curves.csv, km_at_risk.csv.

# Kaggle no longer runs R kernels; this script is run by Rscript from the Python launcher hn-12-revsurvey.py.
# Packages come as Ubuntu binaries from the Posit package manager (CRAN snapshot), falling back to CRAN source.
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
for (p in c("survey", "jsonlite", "mice", "mitools")) if (!requireNamespace(p, quietly = TRUE))
  tryCatch(install.packages(p, repos = "https://packagemanager.posit.co/cran/__linux__/noble/latest"),
           error = function(e) install.packages(p, repos = "https://cloud.r-project.org"))
suppressMessages({library(survey); library(survival); library(jsonlite); library(mice); library(mitools)})
options(survey.lonely.psu = "adjust", width = 200)

f <- list.files("/kaggle/input", pattern = "hn11_analysis.csv.gz", recursive = TRUE, full.names = TRUE)[1]
d <- read.csv(gzfile(f), stringsAsFactors = FALSE, na.strings = c("NA", ""))
cat(R.version.string, " survey", as.character(packageVersion("survey")), " mice", as.character(packageVersion("mice")), "\n")
cat("rows:", nrow(d), " cohort:", sum(d$cohort), "\n")

SCORED <- c("A_reference", "B_painless_PN", "C_painful_PN", "D_hidden", "E_backpain_ctrl", "F_hands_only")
G <- c(B = "B_painless_PN", C = "C_painful_PN", D = "D_hidden", E = "E_backpain_ctrl", F = "F_hands_only")
for (k in names(G)) {
  d[[paste0("g", k)]] <- as.integer(d$group == G[[k]])
  d[[paste0("s", k)]] <- as.integer(d$group_strict == G[[k]])
}
d$race  <- relevel(factor(d$race), ref = "NHWhite")
d$smoke <- relevel(factor(d$smoke), ref = "never")
d$educ  <- relevel(factor(d$educ), ref = "College")
d$t <- d$PERMTH_EXAM / 12
d$e <- as.integer(d$MORTSTAT == 1)
# normal circulation: ABI 1.0-1.4 in every measured leg (both legs when both were measured) and no claudication;
# missing ABI -> NA (left out of the interaction model); missing claudication answers count as no claudication
d$normcirc <- as.integer(d$abi_min >= 1.0 & d$abi_max <= 1.4 & d$claud == 0)
# SMN with isolated foot symptoms: feet only (not hands and feet) and no low back pain
d$smn_isolated <- as.integer(d$group == "D_hidden" & d$sx_feet_only == 1 & d$back_pain == 0)
d$period <- ifelse(d$SDDSRVYR %in% c(1, 2), "1999-2002", "2003-2004")

des <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w6, nest = TRUE, data = d)
DF <- degf(des)
cat("strata:", length(unique(d$SDMVSTRA)), " PSUs:", nrow(unique(d[, c("SDMVSTRA", "SDMVPSU")])), " design df:", DF, "\n")
OUT <- list(versions = list(R = R.version.string, survey = as.character(packageVersion("survey")), mice = as.character(packageVersion("mice"))), design = list(strata = length(unique(d$SDMVSTRA)), psu = nrow(unique(d[, c("SDMVSTRA", "SDMVPSU")])), df = DF))
qt975 <- qt(0.975, DF)
est_ci <- function(est, se, df = DF) {est <- as.numeric(est); se <- as.numeric(se); list(est = est, se = se, lo = est - qt(0.975, df) * se, hi = est + qt(0.975, df) * se,
                                          p = 2 * pt(-abs(est / se), df), df = df)}
# design df of a subpopulation: PSUs minus strata among strata that contain subpopulation members
sub_df <- function(ok) {u <- unique(d[ok, c("SDMVSTRA", "SDMVPSU")]); s <- unique(u$SDMVSTRA)
  nrow(unique(d[d$SDMVSTRA %in% s, c("SDMVSTRA", "SDMVPSU")])) - length(s)}
# design-based (Taylor) variance of a weighted Cox fit from its weighted dfbeta residuals, PSUs within strata;
# used only when svycoxph cannot compute its Wald test (more coefficients than design df, single-cycle analyses)
cox_design_var <- function(fml, ok) {
  dz <- d[ok, ]; dz$wn <- dz$w6 / mean(dz$w6)
  environment(fml) <- environment()
  cf <- coxph(fml, data = dz, weights = wn, model = TRUE, control = coxph.control(iter.max = 100))
  D <- residuals(cf, type = "dfbeta", weighted = TRUE)
  print(round(cbind(coef = coef(cf), se_design = sqrt(diag(crossprod(D)))), 4))
  U <- rowsum(D, paste(dz$SDMVSTRA, dz$SDMVPSU))
  key <- unique(d[d$SDMVSTRA %in% unique(dz$SDMVSTRA), c("SDMVSTRA", "SDMVPSU")])
  Uf <- matrix(0, nrow(key), ncol(D)); rownames(Uf) <- paste(key$SDMVSTRA, key$SDMVPSU)
  Uf[rownames(U), ] <- U
  V <- matrix(0, ncol(D), ncol(D))
  for (h in unique(key$SDMVSTRA)) {
    Uh <- Uf[key$SDMVSTRA == h, , drop = FALSE]; nh <- nrow(Uh)
    if (nh < 2) next
    C <- sweep(Uh, 2, colMeans(Uh)); V <- V + nh / (nh - 1) * crossprod(C)
  }
  list(coef = coef(cf), se = sqrt(diag(V)), names = names(coef(cf)))
}
coh <- d$cohort == 1

# ---------------------------------------------------------------- prevalence and population counts
prev <- list()
dc <- des[coh, ]
for (g in SCORED) {
  ind <- as.formula(paste0("~I(group == '", g, "')"))
  pr <- svyciprop(ind, dc, method = "logit", df = DF)
  tot <- svytotal(ind, dc)
  tt <- coef(tot)[2]; ts <- SE(tot)[2]
  prev[[g]] <- list(n = sum(coh & d$group == g), prop = as.numeric(pr), lo = attr(pr, "ci")[1], hi = attr(pr, "ci")[2],
                    total = unname(tt), total_lo = unname(tt - qt975 * ts), total_hi = unname(tt + qt975 * ts))
}
fs <- des[coh & d$group %in% c("C_painful_PN", "D_hidden"), ]
pr <- svyciprop(~I(group == "D_hidden"), fs, method = "logit", df = DF)
prev$share_of_foot_symptoms <- list(prop = as.numeric(pr), lo = attr(pr, "ci")[1], hi = attr(pr, "ci")[2])
OUT$prevalence <- prev
# pain at other sites among SMN adults (survey-weighted) and MPQ070 missingness, for the multi-site pain question
dd <- des[coh & d$group == "D_hidden", ]
wp <- function(v, dz) {pr <- svyciprop(as.formula(paste0("~I(", v, ")")), dz, method = "logit", df = DF)
  list(prop = as.numeric(pr), lo = attr(pr, "ci")[1], hi = attr(pr, "ci")[2])}
OUT$smn_pain_sites <- list(n = sum(coh & d$group == "D_hidden"),
  back_pain = wp("back_pain == 1", dd), hands_too = wp("sx_hands_and_feet == 1", dd),
  back_and_hands = wp("back_pain == 1 & sx_hands_and_feet == 1", dd), isolated = wp("smn_isolated == 1", dd),
  n_isolated = sum(coh & d$smn_isolated == 1),
  hand_group_back_pain = wp("back_pain == 1", des[coh & d$group == "F_hands_only", ]),
  insensate_nosx_back_or_hand = wp("n_pain_sites >= 1", des[coh & d$group == "B_painless_PN", ]),
  mpq070_missing_cohort = sum(coh & d$mpq070_missing == 1),
  mpq070_missing_by_group = as.list(table(d$group[coh & d$mpq070_missing == 1])))
print(str(OUT$smn_pain_sites))
print(str(prev$D_hidden))

# ---------------------------------------------------------------- score contrasts (scores treated as fixed)
contrast <- function(score, mask = rep(TRUE, nrow(d)), dummies = "g", extra = "") {
  ok <- coh & mask & !is.na(d[[score]]) & !is.na(d$race) &
        (if (extra != "") complete.cases(d[, strsplit(gsub(" ", "", extra), "\\+")[[1]][-1], drop = FALSE]) else TRUE) &
        (if (dummies == "g") d$group %in% SCORED else d$group_strict %in% SCORED)
  dd <- paste0(dummies, names(G))
  fit <- svyglm(as.formula(paste(score, "~", paste(dd, collapse = " + "), "+ age + female + race", extra)), design = des[ok, ])
  b <- coef(fit); s <- SE(fit); df <- sub_df(ok)
  res <- list(n = sum(ok), df = df)
  for (k in names(G)) res[[paste0(k, "_vs_A")]] <- est_ci(unname(b[paste0(dummies, k)]), unname(s[paste0(dummies, k)]), df)
  cv <- setNames(c(1, -1), paste0(dummies, c("D", "E")))
  sc <- svycontrast(fit, cv); res$D_minus_E <- est_ci(unname(coef(sc)), unname(SE(sc)), df)
  cv <- setNames(c(1, -1), paste0(dummies, c("D", "F")))
  sc <- svycontrast(fit, cv); res$D_minus_F <- est_ci(unname(coef(sc)), unname(SE(sc)), df)
  ex <- parse(text = sprintf("(%sD - %sE) / %sB", dummies, dummies, dummies))[[1]]
  sc <- svycontrast(fit, list(pct = ex)); res$DE_over_BA <- est_ci(unname(coef(sc)), unname(SE(sc)), df)
  res$n_by_group <- as.list(table(d$group[ok]))
  res
}
SP <- list(
  main = list("s_main"), uniform = list("s_main_uniform"), nogly = list("s_nogly"),
  nobody = list("s_nobody"), nobody_nogly = list("s_nobody_nogly"),
  adj_body = list("s_main", rep(TRUE, nrow(d)), "g", "+ bmi + waist"),
  nodm = list("s_nodm", d$diabetes == 0), lr = list("s_lr"), en = list("s_en"),
  heldout = list("s_heldout", d$SDDSRVYR == 3), corroborated = list("s_main", rep(TRUE, nrow(d)), "s"),
  sciatica = list("s_sciatica"), lowb12 = list("s_lowb12"), vascular = list("s_vascular"), comorb = list("s_comorb"),
  sciatica_lowb12_vascular = list("s_sciatica_lowb12_vascular"), old_all = list("s_old_all"),
  urate = list("s_urate"), neurotox = list("s_neurotox"), hcv = list("s_hcv"), thyroid = list("s_thyroid"),
  new = list("s_new"), everything = list("s_everything"),
  design = list("s_design"), nodemo = list("s_nodemo"),
  isolated = list("s_main", !(d$group == "D_hidden" & d$smn_isolated == 0)))
OUT$contrasts <- list()
for (nm in names(SP)) {
  a <- SP[[nm]]
  r <- do.call(contrast, c(list(score = a[[1]]), if (length(a) > 1) list(mask = a[[2]]), if (length(a) > 2) list(dummies = a[[3]]),
                           if (length(a) > 3) list(extra = a[[4]])))
  OUT$contrasts[[nm]] <- r
  cat(sprintf("%-26s n=%5d  D-E %.3f (%.3f, %.3f)  D-A %.3f\n", nm, r$n, r$D_minus_E$est, r$D_minus_E$lo, r$D_minus_E$hi,
              r$D_vs_A$est))
}

# ---------------------------------------------------------------- mortality (svycoxph)
mort_ok <- coh & d$ELIGSTAT == 1 & d$PERMTH_EXAM > 0 & d$group %in% SCORED
cc <- complete.cases(d[, c("age", "female", "race", "pir", "smoke", "educ", "bmi", "diabetes", "egfr")])
COV <- "age + female + race + pir + smoke + educ + bmi + diabetes + egfr"
cox <- function(mask = rep(TRUE, nrow(d)), dummies = "g", event = d$e, extra = "", nodm = FALSE, label = "") {
  des2 <- update(des, ev = event)
  grp_ok <- if (dummies == "g") d$group %in% SCORED else d$group_strict %in% SCORED
  ok <- mort_ok & cc & mask & grp_ok
  if (extra != "") ok <- ok & complete.cases(d[, strsplit(gsub(" ", "", extra), "\\+")[[1]][-1], drop = FALSE])
  cov <- if (nodm) sub("\\+ diabetes ", "", COV) else COV
  dd <- paste0(dummies, names(G))
  fml <- as.formula(paste("Surv(t, ev) ~", paste(dd, collapse = " + "), "+", cov, extra))
  df <- sub_df(ok)
  fit <- tryCatch(svycoxph(fml, design = des2[ok, ]), error = function(e) NULL)
  method <- "svycoxph"
  if (is.null(fit)) {
    d$ev <<- event
    m <- cox_design_var(fml, ok); b <- setNames(m$coef, m$names); s <- setNames(m$se, m$names); method <- "dfbeta_taylor"
  } else {b <- coef(fit); s <- SE(fit)}
  res <- list(n = sum(ok), deaths = sum(event[ok]), df = df, method = method,
              A = list(n = sum(ok & d$group == "A_reference"), deaths = sum(ok & d$group == "A_reference" & event == 1)))
  for (k in names(G)) {
    z <- est_ci(unname(b[paste0(dummies, k)]), unname(s[paste0(dummies, k)]), df)
    res[[k]] <- list(HR = exp(z$est), lo = exp(z$lo), hi = exp(z$hi), p = z$p,
                     n = sum(ok & d[[paste0(dummies, k)]] == 1), deaths = sum(ok & d[[paste0(dummies, k)]] == 1 & event == 1))
  }
  if (label != "") cat(sprintf("%-26s n=%5d deaths=%4d  HR_D %.2f (%.2f, %.2f)  HR_E %.2f\n", label, res$n, res$deaths,
                               res$D$HR, res$D$lo, res$D$hi, res$E$HR))
  attr(res, "fit") <- fit
  res
}
M <- list()
main <- cox(label = "main"); M$main <- main
M$nodm <- cox(d$diabetes == 0, nodm = TRUE, label = "no diabetes")
M$excl2y <- cox(!(d$MORTSTAT == 1 & d$PERMTH_EXAM < 24), label = "excl deaths <2y")
M$p1999_2002 <- cox(d$period == "1999-2002", label = "1999-2002")
M$p2003_2004 <- cox(d$period == "2003-2004", label = "2003-2004")
M$corroborated <- cox(dummies = "s", label = "corroborated")
for (x in c("sciatica", "lowb12", "vascular", "comorb", "old_all", "urate", "neurotox", "hcv", "thyroid", "new", "everything"))
  M[[x]] <- cox(d[[paste0("x_", x)]] == 0, label = paste("excl", x))
M$sciatica_lowb12_vascular <- cox(d$x_sciatica == 0 & d$x_lowb12 == 0 & d$x_vascular == 0, label = "excl sci+b12+vasc")
M$adj_abi_claud_cvd <- cox(extra = "+ abi_min + claud + cvd_hist", label = "+ABI, claud, CVD")
M$normal_circulation_only <- cox(d$normcirc %in% 1, extra = "+ cvd_hist", label = "normal circulation only")
M$isolated <- cox(!(d$group == "D_hidden" & d$smn_isolated == 0), label = "SMN isolated foot symptoms")
# missing covariates: the complete-case model, the same model without income-to-poverty ratio, and multiple imputation
cc_nopir <- complete.cases(d[, c("age", "female", "race", "smoke", "educ", "bmi", "diabetes", "egfr")])
local({
  cc_old <- cc; COV_old <- COV
  cc <<- cc_nopir; COV <<- sub("pir \\+ ", "", COV)
  M$no_pir <<- cox(label = "without income-to-poverty ratio")
  cc <<- cc_old; COV <<- COV_old
})
CCV <- c("age", "female", "race", "pir", "smoke", "educ", "bmi", "diabetes", "egfr")
OUT$complete_case <- list(eligible = sum(mort_ok), eligible_deaths = sum(mort_ok & d$e == 1),
  complete = sum(mort_ok & cc), complete_deaths = sum(mort_ok & cc & d$e == 1),
  complete_D = sum(mort_ok & cc & d$group == "D_hidden"), eligible_D = sum(mort_ok & d$group == "D_hidden"),
  missing_by_covariate = as.list(sapply(CCV, function(v) sum(mort_ok & is.na(d[[v]])))),
  missing_deaths_by_covariate = as.list(sapply(CCV, function(v) sum(mort_ok & is.na(d[[v]]) & d$e == 1))))
print(str(OUT$complete_case))
CAUSE <- list(cardiovascular = c(1, 5), cancer = 2, other_causes = c(3, 4, 6, 7, 8, 9, 10))
for (cn in names(CAUSE)) M[[paste0("cause_", cn)]] <- cox(event = as.integer(d$e == 1 & d$UCOD %in% CAUSE[[cn]]), label = cn)
# multiple imputation (20 data sets, chained equations with predictive mean matching; imputation model includes the
# group, event indicator and Nelson-Aalen cumulative hazard), Cox model fitted in each with the survey design, Rubin's rules
mi <- d[mort_ok, c("SEQN", CCV, "group", "e", "t")]
mi$H <- nelsonaalen(mi, t, e)
mi$group <- factor(mi$group); mi$smoke <- factor(mi$smoke); mi$educ <- factor(mi$educ)
meth <- make.method(mi); meth[c("SEQN", "group", "e", "t", "H")] <- ""
pm <- make.predictorMatrix(mi); pm[, c("SEQN", "t")] <- 0
imp <- mice(mi, m = 20, method = meth, predictorMatrix = pm, seed = 2026, printFlag = FALSE)
fits <- lapply(seq_len(20), function(i) {
  ci <- complete(imp, i); di <- d
  for (v in CCV) di[match(ci$SEQN, di$SEQN), v] <- ci[[v]]
  di$smoke <- relevel(factor(di$smoke), ref = "never"); di$educ <- relevel(factor(di$educ), ref = "College")
  dsi <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w6, nest = TRUE, data = di)
  svycoxph(as.formula(paste("Surv(t, e) ~ gB + gC + gD + gE + gF +", COV)), design = dsi[mort_ok, ])
})
mc <- MIcombine(fits)
b <- coef(mc); s <- sqrt(diag(vcov(mc)))
M$mi <- list(n = sum(mort_ok), deaths = sum(mort_ok & d$e == 1), df = DF, method = "multiple_imputation_20",
             A = list(n = sum(mort_ok & d$group == "A_reference"), deaths = sum(mort_ok & d$group == "A_reference" & d$e == 1)))
for (k in names(G)) {
  z <- est_ci(unname(b[paste0("g", k)]), unname(s[paste0("g", k)]))
  M$mi[[k]] <- list(HR = exp(z$est), lo = exp(z$lo), hi = exp(z$hi), p = z$p, n = sum(mort_ok & d$group == G[[k]]),
                    deaths = sum(mort_ok & d$group == G[[k]] & d$e == 1))
}
cat(sprintf("%-26s n=%5d deaths=%4d  HR_D %.2f (%.2f, %.2f)\n", "multiple imputation", M$mi$n, M$mi$deaths, M$mi$D$HR, M$mi$D$lo, M$mi$D$hi))
OUT$cox <- lapply(M, function(r) {attr(r, "fit") <- NULL; r})

# interaction between group and normal circulation (adults with an ankle-brachial index)
ok <- mort_ok & cc & !is.na(d$normcirc)
fit <- svycoxph(as.formula(paste("Surv(t, e) ~ (gB + gC + gD + gE + gF) * normcirc +", COV)), design = des[ok, ])
b <- coef(fit)
z0 <- est_ci(unname(b["gD"]), unname(SE(fit)["gD"]))
sc <- svycontrast(fit, c(gD = 1, "gD:normcirc" = 1)); z1 <- est_ci(unname(coef(sc)), unname(SE(sc)))
zi <- est_ci(unname(b["gD:normcirc"]), unname(SE(fit)["gD:normcirc"]))
OUT$circulation_interaction <- list(n = sum(ok), n_eligible_cc = sum(mort_ok & cc),
  n_missing_abi = sum(mort_ok & cc & is.na(d$normcirc)), deaths = sum(ok & d$e == 1),
  abnormal_reasons_D = list(abi_lt_0.9 = sum(ok & d$gD == 1 & d$abi_min < 0.9, na.rm = TRUE),
    abi_0.90_0.99 = sum(ok & d$gD == 1 & d$abi_min >= 0.9 & d$abi_min < 1.0, na.rm = TRUE),
    abi_gt_1.4 = sum(ok & d$gD == 1 & d$abi_max > 1.4, na.rm = TRUE), claudication = sum(ok & d$gD == 1 & d$claud == 1)),
  n_D_normal = sum(ok & d$gD == 1 & d$normcirc == 1),
  n_D_abnormal = sum(ok & d$gD == 1 & d$normcirc == 0),
  HR_D_abnormal_or_claud = list(HR = exp(z0$est), lo = exp(z0$lo), hi = exp(z0$hi), p = z0$p),
  HR_D_normal = list(HR = exp(z1$est), lo = exp(z1$lo), hi = exp(z1$hi), p = z1$p),
  ratio_of_HR = list(HR = exp(zi$est), lo = exp(zi$lo), hi = exp(zi$hi), p_interaction = zi$p))
print(str(OUT$circulation_interaction))

# proportional hazards: Schoenfeld residuals of the weighted main model
ok <- mort_ok & cc
dz <- d[ok, ]
cf <- coxph(as.formula(paste("Surv(t, e) ~ gB + gC + gD + gE + gF +", COV)), data = dz, weights = w6 / mean(w6), robust = TRUE,
            cluster = paste(SDMVSTRA, SDMVPSU))
zph <- cox.zph(cf, transform = "km", terms = FALSE)
tab <- as.data.frame(zph$table)
OUT$schoenfeld <- list(global_p = tab["GLOBAL", "p"], term_p = as.list(setNames(tab$p, rownames(tab))))
print(round(tab, 4))

# E-value (common outcome, VanderWeele & Ding)
ev_fun <- function(hr) {rr <- (1 - 0.5^sqrt(hr)) / (1 - 0.5^sqrt(1 / hr)); if (rr > 1) rr + sqrt(rr * (rr - 1)) else 1}
OUT$evalue <- list(HR = main$D$HR, lo = main$D$lo, E_point = ev_fun(main$D$HR), E_ci = ev_fun(main$D$lo))

# ---------------------------------------------------------------- survey-weighted Kaplan-Meier curves
ok <- mort_ok
km <- svykm(Surv(t, e) ~ group, design = des[ok, ], se = TRUE)
rows <- list(); grid <- seq(0, 20, by = 0.1)
for (g in names(km)) {
  k <- km[[g]]
  idx <- findInterval(grid, k$time); idx[idx == 0] <- NA
  sv <- ifelse(is.na(idx), 1, k$surv[pmax(idx, 1)])
  se <- ifelse(is.na(idx), 0, sqrt(k$varlog[pmax(idx, 1)]))
  rows[[g]] <- data.frame(group = g, t = grid, surv = sv, lo = exp(log(sv) - qt975 * se), hi = pmin(1, exp(log(sv) + qt975 * se)))
}
kmd <- do.call(rbind, rows); write.csv(kmd, "km_curves.csv", row.names = FALSE)
at <- do.call(rbind, lapply(SCORED, function(g) data.frame(group = g, year = c(0, 5, 10, 15),
  at_risk = sapply(c(0, 5, 10, 15), function(y) sum(ok & d$group == g & d$t >= y)))))
write.csv(at, "km_at_risk.csv", row.names = FALSE)
OUT$km_17y <- lapply(split(kmd[abs(kmd$t - 17) < 1e-6, ], kmd$group[abs(kmd$t - 17) < 1e-6]),
                     function(r) list(surv = r$surv, lo = r$lo, hi = r$hi))
OUT$km_n <- lapply(setNames(SCORED, SCORED), function(g) list(n = sum(ok & d$group == g), deaths = sum(ok & d$group == g & d$e == 1)))

write_json(OUT, "hn12_results.json", auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("done\n")

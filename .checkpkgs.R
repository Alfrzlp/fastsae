ip <- installed.packages()[, "Package"]
cat("sae installed:", "sae" %in% ip, "\n")
cat("emdi installed:", "emdi" %in% ip, "\n")
if ("sae" %in% ip) cat("sae version:", as.character(packageVersion("sae")), "\n")
if ("emdi" %in% ip) cat("emdi version:", as.character(packageVersion("emdi")), "\n")
cat("RhpcBLASctl installed:", "RhpcBLASctl" %in% ip, "\n")

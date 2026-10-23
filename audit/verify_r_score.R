# Run the unmodified causalweight ATET/trim/SE expressions with frozen nuisances.
# This isolates the score. It does not rerun MLfunct or the complete package.
# Usage: Rscript audit/verify_r_score.R fixture.csv didcontDMLpanel.R output.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
source(args[2])
nodes <- as.list(body(didcontDMLpanel))[-1]
start <- which(vapply(nodes, function(e) {
  grepl("^param = cbind\\(kernwgtdtreat", paste(deparse(e),collapse=" "))
}, logical(1)))
stopifnot(length(start)==1)
fixture <- read.csv(args[1])
rows <- list()
for (name in unique(fixture$case)) {
  g <- fixture[fixture$case==name,]
  env <- list2env(list(kernwgtdtreat=g$high/sum(g$high),
    kernwgtdcontrol=g$low/sum(g$low), y=g$y,
    param=cbind(g$pd,g$pc,g$mu), trim=g$trim[1], bw=1,
    cluster=NULL, d=seq_len(nrow(g))), parent=globalenv())
  result <- NULL
  for (node in nodes[start:length(nodes)]) result <- eval(node,env)
  rows[[name]] <- data.frame(case=name, R_ATET=result$ATET,
    R_trimmed=result$ntrimmed, R_se=result$se)
}
write.csv(do.call(rbind,rows),args[3],row.names=FALSE)
set.seed(1)
idx <- sample(12,replace=FALSE)
old_overlap <- vapply(1:3,function(i) {
  test <- idx[((i-1)*4+1):(i*4)]
  length(intersect(test,idx[-test]))
},integer(1))
cat("R version:",R.version.string,"\n")
cat("2024 training/test overlap per fold:",old_overlap,"\n")
cat("2024 prediction order remains permuted:",!identical(idx[sort(idx)],seq_len(12)),"\n")

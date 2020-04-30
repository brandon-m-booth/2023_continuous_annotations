#!/usr/bin/env Rscript
library("optparse");
library("irr");
library(data.table);

option_list = list(
  make_option(c("-i", "--input_csv"), type="character", help="Path to csv containing the signals (rows are time, columns are signals)"),
  make_option(c("-m", "--method"), type="character", help="Method: nominal, ordinal, interval, ratio"),
  make_option(c("-o", "--output_csv"), type="character", help="Path to output csv containing ICC results")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$input_csv)) {
  print_help(opt_parser)
  stop("An input argument must be supplied", call.=FALSE)
}

if (is.null(opt$output_csv)) {
  print_help(opt_parser)
  stop("An output argument must be supplied", call.=FALSE)
}

df = read.table(opt$input_csv, sep=",", header=TRUE);
dft = transpose(df);
dft_mat = as.matrix(dft)
alpha_results = kripp.alpha(dft_mat, opt$method);
write.table(alpha_results$value, opt$output_csv, col.names=c("alpha"), row.names=FALSE)
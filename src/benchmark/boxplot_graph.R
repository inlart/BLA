library(ggplot2)
library(RColorBrewer)
library(gtools)
library(reshape2)

read_data_file <- function(datfile) {
    df = read.table(datfile, header = FALSE)
    colnames(df) <- c("matrix_size", "num_workers", "type", "time_ms")
    return(df)
}

swap_str_order <- function(s) {
    e = strsplit(s, "#")[[1]]
    return(paste(e[2], e[1], sep="#"))
}


generate_factorname <- function(rowlist) {
    type = rowlist[3]
    matrix_size = rowlist[1]
    num_workers = as.numeric(rowlist[2])
    if(num_workers == 0) {
        return(paste("auto", type, sep="#"))
    }
    return(paste(num_workers, type, sep="#"))
}

get_factorname_colour <- function(factorname) {
    factorname = swap_str_order(factorname)
    determinators = strsplit(factorname, "#")[[1]]
    skip_first_n_colours = 4 #dont use the very light colours of the pallete generator
    num_workers_values = unique(sort(df$num_workers))
    if(num_workers_values[1] == 0) {
        num_workers_values = c(tail(num_workers_values, n=-1), 0)
    }
    if(determinators[2] == "auto") {
        our_worker_index = which(num_workers_values == 0) + skip_first_n_colours
    } else {
        our_worker_index = which(num_workers_values == as.numeric(determinators[2])) + skip_first_n_colours
    }
    if(determinators[1] == "ALLSCALE") {
        return(brewer.pal(length(num_workers_values) + skip_first_n_colours, "Greens")[our_worker_index])
    } else if(determinators[1] == "EIGEN") {
        return(brewer.pal(length(num_workers_values) + skip_first_n_colours, "Reds")[our_worker_index])
    } else if(determinators[1] == "NEW1") {
        return(brewer.pal(length(num_workers_values) + skip_first_n_colours, "Purples")[our_worker_index])
    } else { #NEW2
        return(brewer.pal(length(num_workers_values) + skip_first_n_colours, "Blues")[our_worker_index])
    }
}

get_factorname_label <- function(factorname) {
    factorname = swap_str_order(factorname)
    determinators = strsplit(factorname, "#")[[1]]
    if(determinators[1] == "EIGEN") {
        return(paste("Eigen t", determinators[2], sep=""))
    } else if(determinators[1] == "ALLSCALE") {
        return(paste("Allscale t", determinators[2], sep=""))
    } else if(determinators[1] == "NEW1") {
        return(paste("New1 t", determinators[2], sep=""))      
    } else { #NEW2
        return(paste("New2 t", determinators[2], sep=""))
    }
}


args <- commandArgs(trailingOnly = TRUE)

datfile = "with_num_workers.dat"
if (length(args) > 0) { #we got datfile name passed
    datfile = args[1]
}

print(paste("Using datfile: ", datfile))

longdf = read_data_file(datfile)

pdf(paste(substr(datfile, 1, nchar(datfile) - 4), ".pdf", sep=""))

longdf["my_factor"] = apply(longdf, 1, generate_factorname)

df = longdf
#df = df[(longdf$num_workers == 0 | df$num_workers == 1 | df$num_workers == 2),]
#df = df[(longdf$num_workers == 0 | df$num_workers == 1 | df$num_workers == 2 | df$num_workers == 4| df$num_workers == 8),]
#df = df[df$matrix_size %% 150 == 0,]
#df = longdf[(longdf$num_workers == 0),]
#df = df[(df$matrix_size <= 300),]

plot_size_x = max(df$matrix_size) - min(df$matrix_size)
factor_names = unlist(lapply(unique(mixedsort(unlist(lapply(df$my_factor, swap_str_order)))), swap_str_order))
#factor_names = unique(mixedsort(df$my_factor))
factor_labels = unlist(lapply(factor_names, get_factorname_label))
factor_colours = unlist(lapply(factor_names, get_factorname_colour))
number_workers = length(unique(sort(df$num_workers)))

p = ggplot(df, aes(x=factor(num_workers), y=time_ms, colour=my_factor)) +
    geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 0.15, outlier.fill="black", lwd=0.15, position=position_dodge(0.8)) + 
    scale_x_discrete(name = "Number Threads", breaks = c(1, 2, 4, 8, 16)) +
    scale_y_log10(name = "Time to Multiply [ms]", breaks = c(1, 10, 100, 1000, 10000)) + expand_limits(y=1) +
    scale_colour_manual(values=setNames(factor_colours, factor_names), labels=setNames(factor_labels, factor_names), breaks=factor_names) +
    labs(title="Matrix Multiplication in Allscale and Eigen",
         subtitle="Matrix Size",
         colour="") + #remove title in legend
    theme(legend.position = "bottom", legend.direction="horizontal", panel.spacing = unit(0.15, "lines"), plot.subtitle = element_text(hjust = 0.5)) +
    guides(colour=guide_legend(nrow=number_workers, byrow=FALSE)) + facet_grid(~matrix_size, switch="y")


print(p)

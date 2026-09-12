#' @title mys: mean years of schooling people with disabilities.
#' @description A dataset containing the mean years of schooling people with disabilities.
#' @format
#' A data frame with 42 rows and 7 variables with 10 domains are non-sampled areas.
#'
#' \describe{
#'   \item{area}{regency municipality}
#'   \item{y}{mean years of schooling people with disabilities}
#'   \item{vardir}{variance sampling from the direct estimator for each area}
#'   \item{rse}{relative standard error (\%)}
#'   \item{x1}{Number of Elementary Schools}
#'   \item{x2}{Number of Junior High Schools}
#'   \item{x3}{Number of Senior High Schools}
#'   \item{n}{Number of eligible samples}
#'   \item{weight}{Weight}
#' }
"mys"

#' @title mys: mean years of schooling people with disabilities disabilities 2016 - 2026.
#' @description A dataset containing the mean years of schooling people with disabilities 2016 - 2026.
#' @format
#' A data frame with 42 rows and 7 variables with 10 domains are non-sampled areas.
#'
#' \describe{
#'   \item{area}{regency municipality}
#'   \item{y}{mean years of schooling people with disabilities}
#'   \item{year}{year}
#'   \item{vardir}{variance sampling from the direct estimator for each area}
#'   \item{rse}{relative standard error (\%)}
#'   \item{x1}{Number of Elementary Schools}
#'   \item{x2}{Number of Junior High Schools}
#'   \item{x3}{Number of Senior High Schools}
#'   \item{n}{Number of eligible samples}
#'   \item{weight}{Weight}
#' }
"mys_panel"

#' @title Example proximity matrix
#' @description A sample proximity matrix for SAE demo.
#' @format A matrix with n rows and n columns
"mys_proxmat"


#' Corn and Soybean Survey and Satellite Data in 12 Iowa Counties
#'
#' Survey and satellite data for corn and soy beans in 12 Iowa counties,
#' originally obtained from the 1978 June Enumerative Survey of the U.S.
#' Department of Agriculture and from LANDSAT satellite observations during
#' the 1978 growing season.
#'
#' This dataset is included for demonstration purposes and is originally
#' provided in the \pkg{sae} package.
#'
#' @usage data(cornsoybean)
#'
#' @format A data frame with 37 observations on the following 5 variables:
#' \describe{
#'   \item{County}{numeric county code.}
#'   \item{CornHec}{reported hectares of corn from the survey.}
#'   \item{SoyBeansHec}{reported hectares of soy beans from the survey.}
#'   \item{CornPix}{number of pixels of corn in the sample segment within county, from satellite data.}
#'   \item{SoyBeansPix}{number of pixels of soy beans in the sample segment within county, from satellite data.}
#' }
#'
#' @source
#' Battese, G.E., Harter, R.M., and Fuller, W.A. (1988).
#' *An Error-Components Model for Prediction of County Crop Areas Using Survey and Satellite Data.*
#' *Journal of the American Statistical Association*, 83, 28–36.
#'
#'
#' @keywords datasets
"cornsoybean"


#' Corn and Soybean Mean Number of Pixels per Segment for 12 Iowa Counties
#'
#' County means of number of pixels per segment of corn and soy beans,
#' from satellite data, for 12 counties in Iowa. The dataset includes
#' population size, sample size, and means of auxiliary variables used in
#' the dataset \code{\link[sae]{cornsoybean}}.
#'
#' This dataset is provided for demonstration purposes and is originally
#' distributed with the \pkg{sae} package.
#'
#' @usage data(cornsoybeanmeans)
#'
#' @format A data frame with 12 observations on the following 6 variables:
#' \describe{
#'   \item{CountyIndex}{numeric county code.}
#'   \item{CountyName}{name of the county.}
#'   \item{SampSegments}{number of sample segments in the county (sample size).}
#'   \item{PopnSegments}{number of population segments in the county (population size).}
#'   \item{MeanCornPixPerSeg}{mean number of corn pixels per segment in the county.}
#'   \item{MeanSoyBeansPixPerSeg}{mean number of soy beans pixels per segment in the county.}
#' }
#'
#' @source
#' Battese, G.E., Harter, R.M., and Fuller, W.A. (1988).
#' *An Error-Components Model for Prediction of County Crop Areas Using Survey and Satellite Data.*
#' *Journal of the American Statistical Association*, 83, 28–36.
#'
#' @keywords datasets
"cornsoybeanmeans"


### Configuration
#############################################################################
# Load Packages
config_req_pkgs = c('kableExtra', 'MASS', 'plotly', 'rmarkdown', 'stringr', 'tidyverse', 'tidyquant')
lapply(config_req_pkgs, require, character.only = TRUE)
options(scipen = 999)

# Data
df = read.csv('D:/product_growth.csv')[,1:9] %>%
  dplyr::filter(time > 40) %>%
  magrittr::set_colnames(c('dt', 'time', 'value3', 'value2', 'value', 'acquisition', 'new_marketing_platform', 'pricing_error', 'jan')) %>%
  dplyr::mutate(dt = as.Date(dt, format = '%m/%d/%Y')) %>%
  dplyr::mutate(partition = ifelse(time > 156, 'Test', 'Train'))

price_error_df = df[df$partition == 'Train' & df$pricing_error == 1,]



# Prediction 1 - with business context
tseries = ts(df[df$partition == 'Train','value3'], frequency = 12)
tseries_decomposed = decompose(tseries)
arimax_model = forecast::Arima(tseries, order = c(1, 1, 0), seasonal = c(1, 0, 0))
pred_df1 = data.frame(time = df[df$partition == 'Test',]$time, value3 = predict(arimax_model, n.ahead = nrow(df[df$partition == 'Test',]))$pred) %>%
  dplyr::mutate(partition = 'Predicted (black box model)')




# Prediction 2 - with business context
train_x_vars = df[df$partition == 'Train', c('acquisition', 'new_marketing_platform', 'pricing_error')] %>% as.matrix()
test_x_vars = df[df$partition == 'Test', c('acquisition', 'new_marketing_platform', 'pricing_error')] %>% as.matrix()
tseries = ts(df[df$partition == 'Train','value3'], frequency = 12)
tseries_decomposed = decompose(tseries)
arimax_model = forecast::Arima(tseries, order = c(1, 1, 0), seasonal = c(1, 0, 0), xreg = train_x_vars)
pred_df2 = data.frame(time = df[df$partition == 'Test',]$time, value3 = predict(arimax_model, n.ahead = nrow(df[df$partition == 'Test',]), newxreg = test_x_vars)$pred)  %>%
  dplyr::mutate(partition = 'Predicted (with business context)')

# Append Predicted Values (1)
test_df = df %>% dplyr::filter(partition == 'Test')
test_pred_df1 = test_df
test_pred_df1$partition = pred_df1$partition
test_pred_df1$value3 = pred_df1$value3


# Append Predicted Values (2)
test_pred_df2 = test_df
test_pred_df2$partition = pred_df2$partition
test_pred_df2$value3 = pred_df2$value3


df = rbind.data.frame(df, test_pred_df1, test_pred_df2) %>%
  dplyr::mutate(data_type = ifelse(partition %in% c('Train', 'Test'), 'Actual', 'Predicted'))

# Get Errors
mape1 = mean(abs(pred_df1$value3 - test_df$value3) / test_df$value3)
mape2 = mean(abs(pred_df2$value3 - test_df$value3) / test_df$value3)

mape1_label = paste0(round(mape1 * 100, 2), '%')
mape2_label = paste0(round(mape2 * 100, 2), '%')





# Plot 1 - Business Context
ggplot(df %>% dplyr::filter(partition %in% c('Train', 'Test', 'Predicted (with business context)')),
       aes(x = dt, y = value3, color = data_type)) +
  theme_bw() +
  geom_rect(data = df[df$partition == 'Train' & df$pricing_error == 1,],
            aes(xmin = min(price_error_df$dt), xmax=max(price_error_df$dt), ymin=10000, ymax=16000),
            fill = rgb(17, 56, 99, maxColorValue = 255),
            color = rgb(17, 56, 99, maxColorValue = 255),
            alpha = 0.02,
            inherit.aes = FALSE) +
  geom_vline(xintercept = min(df[df$acquisition == 1, 'dt']),
             size = 1,
             color = rgb(17, 56, 99, maxColorValue = 255)) +
  geom_vline(xintercept = min(df[df$new_marketing_platform == 1, 'dt']),
             size = 1,
             color = rgb(17, 56, 99, maxColorValue = 255)) +
  
  
  geom_line() +
  geom_point() +
  geom_vline(xintercept = df[df$jan == 1,'dt'],
             linetype = 'dashed',
             color = 'grey') +
  scale_x_date(date_labels = "'%y", date_breaks = '1 year') +
  labs(x = 'Month', y = 'Products in Force', title = 'Monthly Product Growth (with business context)', color = '',
       subtitle = paste0('Mean % Error: ', mape2_label)) +
  scale_y_continuous(labels = scales::comma,
                     limits = c(8000, max(df[df$partition == 'Train',]$value3))) + 
  theme(axis.text = element_text(size = 8),
        legend.position = 'none')





# Plot 2 - No business context
ggplot(df %>% dplyr::filter(partition %in% c('Train', 'Test', 'Predicted (black box model)')),
       aes(x = dt, y = value3, color = data_type)) +
  theme_bw() +
  
  geom_line() +
  geom_point() +
  geom_vline(xintercept = df[df$jan == 1,'dt'],
             linetype = 'dashed',
             color = 'grey') +
  scale_x_date(date_labels = "'%y", date_breaks = '1 year') +
  labs(x = 'Month', y = 'Products in Force', title = 'Monthly Product Growth',
       subtitle = paste0('Mean % Error: ', mape1_label)) +
  scale_y_continuous(labels = scales::comma,
                     limits = c(8000, max(df[df$partition == 'Train',]$value3))) + 
  theme(axis.text = element_text(size = 8),
        legend.position = 'none') 
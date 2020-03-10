
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import pickle
from pyspark import SparkFiles
from pyspark.sql.functions import col, lit, udf, when
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr


spark = SparkSession \
    .builder \
    .appName("abc-pyspark") \
    .enableHiveSupport() \
    .getOrCreate()
    
dfBase = spark.sql("""select * from fdr_sample.test_data_t2p where customer_segment = 1""")
print('ATAC : Data successfully pulled from HIVE table')

rec_cnt = spark.sparkContext.accumulator(0)

dfBase.printSchema()

dfBase = dfBase.withColumn("ratio_closed_0", col("closed_bucket_0")/col("count_paid_invoices"))
dfBase = dfBase.withColumn("ratio_closed_1", col("closed_bucket_1")/col("count_paid_invoices"))
dfBase = dfBase.withColumn("ratio_closed_2", col("closed_bucket_2")/col("count_paid_invoices"))
dfBase = dfBase.withColumn("ratio_closed_3", col("closed_bucket_3")/col("count_paid_invoices"))
dfBase = dfBase.withColumn("ratio_closed_4", col("closed_bucket_4")/col("count_paid_invoices"))

print('ATAC : New Columns succesfully created')

print('ATAC : Old column')
dfBase.select("closed_bucket_1").show(n=5)

print('ATAC : New Column')
dfBase.select("ratio_closed_1").show(n=5)

print('ATAC : Shape of dataframe : ', dfBase.count(), len(dfBase.columns))


# #######
# Load the model from HDFS
spark.sparkContext.addFile('hdfs://CRT01/apollo/edl/finance/trg_file/pickle/Survival_Model_EDL_Data.pkl')
with open(SparkFiles.get('Survival_Model_EDL_Data.pkl'), 'rb') as handle:
    a_model = pickle.load(handle)

print('ATAC : Model Load Successful')



######## Subsetting the columns

dfBase2 = dfBase.select([c for c in dfBase.columns if c in ['invoice_nbr','country_code','invoice_base_amount','commited_days', 'invoice_category', 'dso2',
            'financial_banking_cfi','financial_banking_major', 'financial_channel', 'financial_other_financial_serv','hospitality_channel',
            'hospitality_eating_drinking_qs','hospitality_eating_drinking_ts','hospitality_sports_recreation','retail_channel',
            'retail_cstore_and_petroleum','retail_distribution_wholesale','retail_drug','retail_food_and_mass',
            'retail_public_sector','retail_specialty_retail','t_and_t_technology','tlg_channel',
            'tlg_gaming','tlg_lodging','tlg_travel',
            'month_num_created','day_of_week_created','day_of_month_due','month_num_due','day_of_week_due',
            'ratio_late_paid_amount',
            'ratio_closed_0', 'ratio_closed_1', 'ratio_closed_2', 'ratio_closed_3', 'ratio_closed_4', 
            'hw','sw','swm','hwm','ps','ts','other_category','paid_status']])

# Filling the nulls with 0
dfBase3 = dfBase2.na.fill(0)


print('ATAC : Shape of New dataframe : ', dfBase3.count(), len(dfBase3.columns))    

column_names = dfBase3.columns

#def partition_predictor(input_rows):
#    global a_model
#    global rec_cnt
#    global column_names
#    final_iterator = []
 #   for input_row in input_rows:
 #       rec_cnt += 1
#        df_row = pd.DataFrame([input_row])
       # print('ATAC_Succesful Conversion of row to Pandas DF, Now dropping invoice nbr', df_row)
       # pred_df_row=df_row.drop(['invoice_nbr'], axis=1)
       # print('ATAC_Printing input_row', input_row)
       # print('ATAC_Printing dfRow', df_row)
#        print( df_row.dtypes)
 #       final_iterator.append({ "invoice_nbr" : input_row['invoice_nbr'], "Predicted_T2P" : a_model.predict_median(df_row).tolist()})
#    return final_iterator



def partition_predictor(input_rows):
    global a_model
    global rec_cnt
    global column_names
    print('ATAC Column Names', column_names)
    final_iterator = []
    for input_row in input_rows:
        rec_cnt += 1
        df_row = pd.DataFrame([input_row],columns=column_names)
        print('ATAC_Succesful Conversion of row to Pandas DF, Now dropping invoice nbr', df_row)
        pred_df_row=df_row.drop(['invoice_nbr'],axis=1)
       # print('ATAC_Printing input_row', input_row)
       # print('ATAC_Printing dfRow', df_row)
        final_iterator.append({ "invoice_nbr" : input_row['invoice_nbr'], "Predicted_T2P" : a_model.predict_median(pred_df_row).tolist()})
    return final_iterator



#Calling the functions
print("Input Paritions={}".format(dfBase3.rdd.getNumPartitions()))
preds = dfBase3.rdd.repartition(1000).mapPartitions(partition_predictor).cache()
print("ATAC_NBR Predictions!",preds.count())
preds_df = preds.toDF()
print('Datatype is', type(preds_df))
print('Shape of Spark Dataframe', (preds_df.count(), len(preds_df.columns)))

final_dataset = dfBase.join(preds_df, dfBase['invoice_nbr'] == preds_df['invoice_nbr'], "left").drop(dfBase.invoice_nbr)
print('Final_Dataset_Shape', (final_dataset.count(), len(final_dataset.columns)))

#final_dataset = final_dataset.select([c for c in final_dataset.columns if c not in columns_to_drop])

    
#Summarizing the output

# replacing the infinity values by udf
replace_infs_udf = udf(
    lambda x, v: float(v) if x and np.isinf(x) else x, DoubleType()
)

final_dataset = final_dataset.withColumn("Predicted_T2P", replace_infs_udf(col("Predicted_T2P"), lit(999.0)))
print('ATAC : Replacement of infinity values successful')

# adding the predicted days to invoice date to get predicted date
#final_dataset = final_dataset.withColumn("Predicted_Date", F.date_add(final_dataset["invoice_date"], final_dataset["Predicted_T2P"]))
final_dataset = final_dataset.withColumn("Predicted_Date", expr("date_add(invoice_date, Predicted_T2P)"))
print('ATAC : Adding days to date and creating Predicted Date successful',final_dataset.select('Predicted_Date').show(n=5))

print(final_dataset.select('invoice_nbr','Predicted_T2P','Predicted_Date').show(n=5))
#print(final_dataset.select('invoice_nbr').show(n=5))

final_dataset.describe(['Predicted_T2P']).show()



# Prediction for February
#feb = final_dataset.filter("Predicted_Date >='2020-02-01' AND Predicted_Date <='2020-02-29'")
feb = final_dataset.where((col('Predicted_Date') >= lit('2020-02-01')) & (col('Predicted_Date') <= lit('2020-02-29')))

print('Dataframe shape for Feb :', (feb.count(), len(feb.columns)))
feb.select('invoice_base_amount').show(n=5)
print('Predicted Amount for February:')

import pyspark.sql.functions as F  
feb2 = feb.select([c for c in feb.columns if c in ['invoice_base_amount']])
feb.groupBy().agg(F.sum('invoice_base_amount')).show()

print('Program Ends')

# Writing out as a file
#final_dataset.coalesce(1).write.mode('overwrite').option("header","true").csv('/user/at185217/Predicted_Dataset.csv')

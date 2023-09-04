import pandas as pd
import numpy as np
import glob
import os

class data_loader:
    def getData(self,path):
        df: pd.DataFrame = pd.read_csv(path, encoding='big5', header=None)
        "清除多餘的空白"
        df = df.applymap(lambda x: x.strip())
        df.columns = df.iloc()[0, :]
        df: pd.DataFrame = df.iloc()[2:, :]
        return df
    def convert_columns(self,df: pd.DataFrame):
        # 統計日期
        date_list: list = df.iloc()[:, 1].value_counts().index.to_list()
        # print(f'day = {len(date_list)}')
        df_list = []
        # 按日處理資料
        for date in date_list:
            df1: pd.DataFrame = df[df['日期'] == date].iloc()[:, 3:].T
            df1.columns = df[df['日期'] == date].iloc()[:, 2].to_list()
            df1.index = pd.date_range(start=date, periods=24, freq="H")
            df_list.append(df1)
        # 合併 DataFrame
        df2 = pd.concat(df_list)
        df2 = df2.reset_index()
        df2.rename(columns={'index': 'Date'}, inplace=True)
        df2.index = pd.to_datetime(df2['Date'])
        return df2.sort_index()
    def convert_Nan(self,x):
        xx = str(x)
        if '#' in xx:
            return np.nan
        elif '*' in xx:
            return np.nan
        elif 'x' in xx:
            return np.nan
        elif 'A' in xx:
            return np.nan
        else:
            return x

    def unit_prepro(self,path):
        df = self.getData(path)
        df = self.convert_columns(df) 
        df1 = df.applymap(self.convert_Nan) #remove invalid value
        df2 = df1.iloc()[:, 1:].astype(float, copy=False)
        #df2 = df2.interpolate()
        return df2

    def main(self,path,place,start,end,freq = None):
        final = pd.DataFrame()
        for year in range(start,end+1):
            path_year = os.path.join(path,'{0}_{1}.csv'.format(place,year))
            print(path_year)
            df = self.unit_prepro(path_year)
            final = pd.concat([final,df],axis=0)
        final = final.interpolate()
        if freq != None:
            #final = final.resample(freq).mean() 
            final = final.asfreq(freq)
            final = final.interpolate()
        print('Place: {0}, Data shape: {1}'.format(place,final.shape))
        return final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import boxcox1p
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    """This class is a dataframe selector.
        Data members:
            features: A list of column_names you want in output dataframe
    """
    def __init__(self,features):
        self.features=features
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self.features]
    
class Misc_Feature_Engineering(BaseEstimator,TransformerMixin):
    """
    This class does following feature engineering
    1. Encode 'emp_length'
    2. Change 'house_ownership' values
    3. Calculate 'earlies_cr_line'-'issue_d'
    4. Divide 'annual_inc','tot_cur_bal','total_rev_hi_lim','revol_bal' by 'funded_amnt'
    """
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        from datetime import datetime as dt
        
        def encode_emp_length(x):
            if type(x)==float: #Check for nan ,as only Nans are float while other values are str
                return x
            if x=='10+ years':
                return 10
            if x=='< 1 year':
                return 0
            else:
                return int(x.split(' ')[0])
        
        X.loc[:,'emp_length']=X['emp_length'].apply(encode_emp_length)
        X.loc[:,'home_ownership']=X['home_ownership'].apply(lambda x: 'RENT' if x in ['OTHER','NONE','ANY'] else x)
        
        # replane null with issue_d
        index=X[X['earliest_cr_line'].isnull()].index
        X.loc[index,'earliest_cr_line']=X.loc[index,'issue_d']

        X.loc[:,'issue_d']=X['issue_d'].apply(lambda x:dt.strptime('01-'+x,'%d-%b-%Y').date())
        X.loc[:,'earliest_cr_line']=X['earliest_cr_line'].apply(lambda x:dt.strptime('01-'+x,'%d-%b-%Y').date())
        X.loc[:,'earliest_cr_line']=X['issue_d']-X['earliest_cr_line']
        X.loc[:,'earliest_cr_line']=X['earliest_cr_line'].apply(lambda x:int(str(x).split(' ')[0]))
        X.drop('issue_d',axis=1,inplace=True)
        
        tmp=['annual_inc','tot_cur_bal','total_rev_hi_lim','revol_bal']
        for col in tmp:
            X[col]=X[col]/X['loan_amnt']
        return X

class total_rev_hi_lim_Imputer(BaseEstimator,TransformerMixin):
    """
    This will impute null values in 'total_rev_hi_lim'.
    This class fits a 2-degree polynomial between 'total_rev_hi_lim' and 'revol_bal' and use it to predict
    null values.
    The polynomial is saved as a class variable and can be used to transform test data.
    """
    def __init__(self):
        self._poly=None

    def fit(self,X,y=None):
        from scipy import polyfit,poly1d
        tmp1=X[X['total_rev_hi_lim'].notnull()][['revol_bal','total_rev_hi_lim']].copy()
        p=polyfit(tmp1['revol_bal'],tmp1['total_rev_hi_lim'],deg=2)
        self._poly=poly1d(p)
        return self
    
    def transform(self,X,y=None):
        X['temp']=X['revol_bal'].apply(self._poly)
        i=X[X['total_rev_hi_lim'].isnull()].index
        X.loc[i,'total_rev_hi_lim']=X.loc[i,'temp']
        X.drop('temp',axis=1,inplace=True)
        return X    

class Median_Imputer(BaseEstimator,TransformerMixin):
    """
    This class will impute given columns with median values.
    It stores the median values of that features which can be further used to transform test data
    
    param:
        columns_to_impute=list, list of columns to impute
    """
    def __init__(self,columns_to_impute):
        self._columns=columns_to_impute
        self._median_dict={}
    
    def fit(self,X,y=None):
        for col in self._columns:
            self._median_dict[col]=X[col].median()
        return self
    
    def transform(self,X,y=None):
        for col in self._columns:
                X.loc[:,col]=X[col].apply(lambda x: self._median_dict[col] if np.isnan(x) else x)
        return X
    
class Dummy_Variables(BaseEstimator,TransformerMixin):
    """
    This class will create dummy variables for categorical columns as well as states from 'addr_state' column
    
    param:
        states_to_keep:list, list of states for which dummy variables are to be created
        cat_cols: list, list of categorical columns for which dummy variables are to be created
    """
    def __init__(self,states_to_keep=[],cat_cols=[]):
        self._states_to_keep=states_to_keep
        self._cat_cols=cat_cols
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for i in self._states_to_keep:
            X.loc[:,'state_'+i]=X['addr_state'].apply(lambda x: 1 if x==i else 0)
        X.drop('addr_state',axis=1,inplace=True)
        
        for col in self._cat_cols:
            X=X.join(pd.get_dummies(X[col],drop_first=True))
        X.drop(self._cat_cols,axis=1,inplace=True)
        return X
    
class Transform_Skewed_Features(BaseEstimator,TransformerMixin):
    """
    This class will transform skewed features in the dataset.
    Feature names as well as tranformations are hard coded in this class.
    """
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X.loc[:,'delinq_2yrs']=X['delinq_2yrs'].apply(lambda x : 5.0 if x > 5 else x)
        X.loc[:,'inq_last_6mths']=X['inq_last_6mths'].apply(lambda x: 4.0 if x>4 else x)
        X.loc[:,'open_acc']=X['open_acc'].apply(lambda x: 2.0 if x in [0.0,1.0] else x)
        X.loc[:,'open_acc']=X['open_acc'].apply(lambda x: 25.0 if x>25 else x)
        X.loc[:,'pub_rec']=X['pub_rec'].apply(lambda x: 4.0 if x>4 else x)
        X.loc[:,'acc_now_delinq']=X['acc_now_delinq'].apply(lambda x: 1.0 if x>1 else x)

        X.loc[:,'int_rate']=X['int_rate'].apply(lambda x: np.log1p(x)**2)
        X.loc[:,'annual_inc']=X['annual_inc'].apply(lambda x:boxcox1p(x,-0.7))
        #X=X[X['dti']<40].reset_index(drop=True)
        X.loc[:,'dti']=X['dti'].apply(lambda x: 40 if x>40 else x)
        X.loc[:,'revol_bal']=X['revol_bal'].apply(lambda x: np.log1p(x)**0.35)
        X.loc[:,'revol_util']=X['revol_util'].apply(lambda x: 140 if x>140 else x)
        X.loc[:,'total_acc']=X['total_acc'].apply(lambda x: np.log1p(x)**1.5)
        X.loc[:,'tot_coll_amt']=X['tot_coll_amt'].apply(lambda x: 80000 if x>80000 else x)
        X.loc[:,'tot_cur_bal']=X['tot_cur_bal'].apply(lambda x: np.log1p(x)**0.75)
        X.loc[:,'total_rev_hi_lim']=X['total_rev_hi_lim'].apply(lambda x: np.log1p(x)**0.5)
        return X

#7 Multicoliniarity check, p-value check
# Train-Test split
def remove_by_pvalue(X,y,pvalue=0.05):
    """Remove features with p-value more than 'pvalue'
    
    This function uses statsmodels.api.Logit model. Please add intercept to data externally.
    Input:
        X: Array or dataframe excluding predicted variable
        y: Series or list of predicted variable
        pvalue: int or float
    
    Note:
        X is changed inplace
    """
    import statsmodels.api as sm
    for i in range(len(X.columns)):
        regressor_Logit=sm.Logit(endog=y,exog=X).fit()
        s=regressor_Logit.pvalues.sort_values(ascending=False)
        if s.iloc[0]>pvalue:
            X.drop(s.index[0],axis=1,inplace=True)
            print('Removed: ',s.index[0],'P-value: ',s.iloc[0])


def remove_by_vif(X,vif=5):
    """Remove columns from X whose VIF is greater than supplied 'vif'
    Parameters:
        X:array or dataframe containing data excluding target variable
        vif: int or float of limiting value of VIF
    Note:
        This function changes X inplace
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
        
    for i in range(len(X.columns)):
        l = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        s=pd.Series(index=X.columns,data=l).sort_values(ascending=False)
        if s.iloc[0]>vif:
            X.drop(s.index[0],axis=1,inplace=True)
            print('Removed: ',s.index[0],', VIF: ',s.iloc[0])
        else:
            break

def create_performance_chart(data,y_pred,y_test):
    picked=data.loc[y_pred==0]
    grades=['A','B','C','D','E','F','G']
    tmp=pd.DataFrame(columns=['ROI','% Picked','% Default'],index=grades)
    for g in grades:
        grade_picked=picked[picked['grade']==g]
        if len(grade_picked.index)==0:
            roi=None
            default_rate=None
            perc_picked=0
        else:
            num_picked=len(grade_picked.index)
            num_total=len(data[data['grade']==g].index)
            roi=sum(grade_picked['total_pymnt'])/sum(grade_picked['funded_amnt'])-1
            l=y_test[list(grade_picked.index)]
            default_rate=100*len(l[l==1])/len(l)
            perc_picked=100*num_picked/num_total

        tmp.loc[g,'% Picked']=perc_picked
        tmp.loc[g,'ROI']=roi
        tmp.loc[g,'% Default']=default_rate
    return tmp
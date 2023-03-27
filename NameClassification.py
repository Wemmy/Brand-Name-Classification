import re
from gensim import corpora
import logging
import pandas as pd
from gensim import models
import pickle
from gensim import similarities
from collections.abc import Sequence
import openpyxl
def make_tuples(l):
    return tuple(make_tuples(i) if isinstance(i, Sequence) else i for i in l)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


unit_list = ['gm', 'g','gram','lb', 'gms', 'pounds', 'lbs', 'gr', 
             'kg', 'mg', 'pkg','pk', 'ea', 'each', 'pcs', 'pc', 'btl', 
                'bottle','sl','cg', 'can', 'lt','sl', 'litre','liter', 
                'loz','ml','oz', 'fl', 'ltr', 'l']

## data cleaning
def clean_size(doc):
    # lowercase the unit, 
    doc = doc.lower()
    # replace all special character with space
    doc = re.sub('[^a-zA-Z0-9\s]', ' ', doc)
    # remove the space before unit
    doc = re.sub(r'(\d+)(\s+)('+'|'.join(unit_list)+')(\s+|$)',r'\1\3\4', doc)
    # normalize the unit
    doc = re.sub(r'(\d+)(gr|gm|gram|gms)(\s+|$)',r'\1g\3', doc)
    doc = re.sub(r'(\d+)(lt|liter|litre|ltr)(\s+|$)',r'\1l\3', doc)
    doc = re.sub(r'(\d+)(pounds|lbs)(\s+|$)',r'\1lb\3', doc)
    doc = re.sub(r'(\d+)(pkg)(\s+|$)',r'\1pk\3', doc)
    doc = re.sub(r'(\d+)(each)(\s+|$)',r'\1ea\3', doc)
    doc = re.sub(r'(\d+)(pcs)(\s+|$)',r'\1pc\3', doc)
    doc = re.sub(r'(\d+)(bottle)(\s+|$)',r'\1btl\3', doc)
    return doc

def split_size(doc):
    '''
    retrun a list: [
                    cleand item: str
                    size: str
                    ]
    '''
    item = re.sub(r'\d+(?:'+ '|'.join(unit_list) +')(?:\s+|$)','', doc).strip()
    size = re.findall(r'\d+(?:'+ '|'.join(unit_list) +')(?:\s+|$)', doc)
    if len(size) > 1:
        size = [size[0]]
    if not size:
        size = ['']
    return [item] + size

def special_cases(doc):
    doc = re.sub('\s+root beer\s+','rootbeer', doc)
    doc = re.sub('\s+choc\s+','chocolate', doc)
    doc = re.sub('\s+snapd\s+','snapped', doc)
    doc = re.sub('\s+razzle berry\s+','razzleberry', doc)
    return doc

def data_cleaning(doc):
    return split_size(special_cases(clean_size(doc)))

def get_max_index(singe_index_list):
    max_sim = singe_index_list[0][1]
    list_index_sim = list(zip(*singe_index_list))
    list_index_max = []
    for i,s in enumerate(list_index_sim[1]):
        if s == max_sim:
            list_index_max.append(list_index_sim[0][i])
    return list_index_max

def read_table(file_name: str, table_name: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(file_name, read_only= False, data_only = True) # openpyxl does not have table info if read_only is True; data_only means any functions will pull the last saved value instead of the formula
    for sheetname in wb.sheetnames: # pulls as strings
        sheet = wb[sheetname] # get the sheet object instead of string
        if table_name in sheet.tables: # tables are stored within sheets, not within the workbook, although table names are unique in a workbook
            tbl = sheet.tables[table_name] # get table object instead of string
            tbl_range = tbl.ref #something like 'C4:F9'
            break # we've got our table, bail from for-loop
    data = sheet[tbl_range] # returns a tuple that contains rows, where each row is a tuple containing cells
    content = [[cell.value for cell in row] for row in data] # loop through those row/cell tuples
    header = content[0] # first row is column headers
    rest = content[1:] # every row that isn't the first is data
    df = pd.DataFrame(rest, columns = header)
    wb.close()
    return df

class MyCorpus:
    def __init__(self,  train_path='2022-11 Beverage & Snacks CPG Price List v3 (2).xlsx', test_path=None) -> None:
        self.train_path = train_path
        self.test_path = test_path
        # self.df_train = pd.read_excel(self.train_path)
        self.df_train = read_table(self.train_path, table_name = 'Table1')
        self.df_train[['Item_WinterList','Size_WinterList']]  = self.df_train['Item Description'].apply(lambda x: pd.Series(data_cleaning(x)))
        self.corpus = []
        self.size = []
        self.bow = []
        self.index = None
        self.dictionary = None
        self.model = None
        self.df_pos_bow = None
        self.df_pos_index = None

    # for strea data: current data are small enouth to be fully cached
    # def __iter__(self): 
    #     # for line in open('E:\Work\Winter2023PricingTracking\Data\df_pos_menuitemname.csv', 'r'):
    #     for line in open(self.test_path, 'r'):
    #         # assume there's one document per line, tokens separated by whitespace
    #         # doc = data_cleaning(line.strip())
    #         yield line.strip()   

    def create_dictionary(self, stopwords = ['plus','regular']):
        texts = [[word for word in row.split() if word not in stopwords] for row in self.df_train['Item_WinterList']]
        self.dictionary = corpora.Dictionary(texts)
    
    def create_model(self):
        corpus_train =[self.dictionary.doc2bow(text) for text in [[word for word in row.split()] for row in self.df_train['Item_WinterList']]]
        self.model = models.LsiModel(corpus_train, id2word=self.dictionary, num_topics=466)
        self.index = similarities.MatrixSimilarity(self.model[corpus_train])

    def save_corpus(self, path = 'E:\Work\Winter2023PricingTracking\Data\mycorpus_pos.pkl'):
        d = {'Corpus': self.corpus, 'Size': self.size, 'Bow': self.bow, 'Index': self.index, 'Dictionary': self.dictionary, 'Model':self.model,
             'Bow_Test': self.df_pos_bow, 'Index_Test':self.df_pos_index}
        with open(path, 'wb') as f:
            pickle.dump(d, f)
    
    def load_corpus(self, path = 'E:\Work\Winter2023PricingTracking\Data\mycorpus_pos.pkl'):
        with open(path, 'rb') as f:
            d = pickle.load(f)
            self.corpus, self.size, self.bow, self.index, self.dictionary, self.model = d['Corpus'], d['Size'], d['Bow'], d['Index'], d['Dictionary'], d['Model']
            self.df_pos_bow, self.df_pos_index = d['Bow_Test'], d['Index_Test']
    
    def load_test_data(self, path = 'E:\Work\Winter2023PricingTracking\Data\df_MenuItemWithPrice&Scancode_Pos.csv'):
        self.df_pos = pd.read_csv(path, names = ['Name', 'Price', 'Scancode'])
        self.df_pos = self.df_pos.fillna('')
        self.df_pos[['Name_Pos','Size_Pos']]  = self.df_pos['Name'].apply(lambda x: pd.Series(data_cleaning(x)))  
    
    # calculate index of df_test
    def create_index_text(self, stopwords = ['plus','regular']):
        self.load_test_data()
        texts = [[word for word in row.split() if word not in stopwords] for row in self.df_pos['Name_Pos']]
        self.df_pos_bow = [self.dictionary.doc2bow(text) for text in [[word for word in row.split()] for row in self.df_pos['Name_Pos']]]
        self.df_pos_index = [sorted(enumerate(self.index[self.model[doc]]), key=lambda item: -item[1])[:10] for doc in self.df_pos_bow]
    
    def construct_test_dataframe(self):
        self.new_data_frame = pd.DataFrame(columns= ['Name', 'Name_Pos','Size_Pos','Price_Pos', 'Scancode', 'Name_WinterList', 'Item_WinterList', 'Size_WinterList', 'Price_WinterList', 'UPCcode','Cos_Sim' ])
        index_list = [get_max_index(e) if e[0][1]!=0 else None for e in self.df_pos_index]
        sim_list = [e[0][1] if e[0][1]!=0 else None for e in self.df_pos_index]

        for i,e in enumerate(index_list):
            if e:
                for idx in e:
                    add_on = {'Name': self.df_pos['Name'][i], 
                            'Name_Pos':self.df_pos['Name_Pos'][i],
                            'Size_Pos':self.df_pos['Size_Pos'][i],
                            'Price_Pos':self.df_pos['Price'][i], 
                            'Scancode':self.df_pos['Scancode'][i],
                            'Name_WinterList':self.df_train['Item Description'][idx],
                            'Item_WinterList':self.df_train['Item_WinterList'][idx], 
                            'Size_WinterList':self.df_train['Size_WinterList'][idx], 
                            'Price_WinterList':self.df_train['Winter 2023 SRP'][idx],
                            'UPCcode':self.df_train['UPC Code'][idx],
                            'Cos_Sim': sim_list[i]}
                    self.new_data_frame = self.new_data_frame.append(add_on, ignore_index=True)
            else:
                    add_on = {'Name': self.df_pos['Name'][i], 
                            'Name_Pos':self.df_pos['Name_Pos'][i],
                            'Size_Pos':self.df_pos['Size_Pos'][i],
                            'Price_Pos':self.df_pos['Price'][i], 
                            'Scancode':self.df_pos['Scancode'][i],
                            'Name_WinterList':None,
                            'Item_WinterList':None, 
                            'Size_WinterList':None, 
                            'Price_WinterList':None,
                            'UPCcode':None,
                            'Cos_Sim': None}
                    self.new_data_frame = self.new_data_frame.append(add_on, ignore_index=True)

def compare_unit(s1, s2):
    if s1 == s2:
        return True
    if s1 and s2:
        u1 = re.findall('\d+(\D+)', s1)[0] 
        u2 = re.findall('\d+(\D+)', s2)[0]
        if u1 == u2:
            return False
        # to ml
        if u1 == 'oz' and u2 == 'g':
            num_size_1 = int(re.findall('(\d+)oz', s1)[0])
            num_size_2 = int(re.findall('(\d+)g', s2)[0])
            if num_size_2*0.9 < num_size_1*28.35 < num_size_2*1.1:
                return True
        if u1 == 'oz' and u2 == 'ml':
            num_size_1 = int(re.findall('(\d+)oz', s1)[0])
            num_size_2 = int(re.findall('(\d+)ml', s2)[0])
            if num_size_2*0.9 < num_size_1*28.41 < num_size_2*1.1:
                return True
    else:
        return False
        
def conditions(s):
    # 1.if we can compare Scancode
    if s['Scancode'] and s['UPCcode']:
        if s['Scancode'] == s['UPCcode']:
            return 1
        else:
            return 0
    # 2. Can not compare Scancode but with similarity larger than 90%
    elif s['Cos_Sim']:
        if s['Cos_Sim'] > 0.9: 
            if s['Size_Pos'] and s['Size_WinterList']:
                # 3. with matched size
                if compare_unit(s['Size_Pos'], s['Size_WinterList']):
                    return s['Cos_Sim']
                # 3. unmatched size
                else:
                    return s['Cos_Sim'] - 0.5 
            else:
                # 4. Can not compare size, but with price
                if  s['Price_WinterList']*0.9  < s['Price_Pos'] < s['Price_WinterList']*1.1:
                    return s['Cos_Sim']
        else:
            # 5. similarity < 0.9
            return 0
    else:
        return 0
    

if __name__ == "__main__":
    mc = MyCorpus()
    mc.create_dictionary()
    mc.create_model()
    mc.create_index_text()
    mc.save_corpus()
    mc.construct_test_dataframe()

    mc.new_data_frame['Confidence'] =mc.new_data_frame.apply(conditions, axis=1)
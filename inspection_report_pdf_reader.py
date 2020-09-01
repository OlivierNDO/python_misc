### Configuration
###############################################################################
import datetime
from io import StringIO
import pandas as pd
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import re


# Input File(s)
# Source: https://vadenpropertyinspections.com/wp-content/uploads/2016/03/Inspection-Report.pdf
config_file_path = 'D:/sample_inspection_report.pdf'





### Define Class
###############################################################################

class InspectionPdfReader:
    """
    Extract text and create features from homeowner inspection report pdf file
    Args:
        pdf_file_path (str): path in operating system to pdf file
    """
    def __init__(self, 
                 pdf_file_path,
                 inspector_name_regex = r'Inspector,\s+\w+\s+\w+',
                 inspection_date_regex = r'INSPECTED ON:\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
                 inspection_date_input_format = '%B %d %Y',
                 inspection_date_output_format = '%Y-%m-%d',
                 checkmark_identifier = '☑'):
        self.pdf_file_path = pdf_file_path
        self.inspector_name_regex = inspector_name_regex
        self.inspection_date_regex = inspection_date_regex
        self.inspection_date_input_format = inspection_date_input_format
        self.inspection_date_output_format = inspection_date_output_format
        self.checkmark_identifier = checkmark_identifier
     
    def get_string(self):
        output = StringIO()
        with open(self.pdf_file_path, 'rb') as f:
            parser_obj = PDFParser(f)
            doc_obj = PDFDocument(parser_obj)
            r_manager = PDFResourceManager()
            device = TextConverter(r_manager, output, laparams = LAParams())
            interpreter = PDFPageInterpreter(r_manager, device)
            for p in PDFPage.create_pages(doc_obj):
                interpreter.process_page(p)
        clean_output = ' '.join(output.getvalue().replace('\n', ' ').split())
        return clean_output
        
    def get_string_list(self):
        output = StringIO()
        with open(self.pdf_file_path, 'rb') as f:
            parser_obj = PDFParser(f)
            doc_obj = PDFDocument(parser_obj)
            r_manager = PDFResourceManager()
            device = TextConverter(r_manager, output, laparams = LAParams())
            interpreter = PDFPageInterpreter(r_manager, device)
            for p in PDFPage.create_pages(doc_obj):
                interpreter.process_page(p)
        clean_output = [s for s in output.getvalue().split('\n') if s != '']
        return clean_output
       
    def get_inspection_date(self):
        pdf_string = self.get_string()
        date_str = re.search(self.inspection_date_regex, pdf_string)
        if date_str is not None:
            reform_date_str = ' '.join(date_str.group().replace(',', '').split(' ')[-3::])
            output = datetime.datetime.strptime(reform_date_str, self.inspection_date_input_format).strftime(self.inspection_date_output_format)
        else:
            output = None
        return output
    
    def get_inspector_name(self):
        pdf_string = self.get_string()
        try:
            inspector_name = re.findall(self.inspector_name_regex, pdf_string)[0].split('Inspector, ')[1]
        except:
            inspector_name = ''
        return inspector_name
    
    def get_client_name(self):
        string_list = self.get_string_list()
        try:
            client_name = [s for s in string_list if 'Prepared For' in s][0].replace('Prepared For:   ', '').rstrip()
        except:
            client_name = ''
        return client_name
    
    def get_client_location(self):
        string_list = self.get_string_list()
        try:
            client_loc = [s for s in string_list if 'Concerning: ' in s][0].replace('Concerning:    ', '').rstrip().lstrip()
        except:
            client_loc = ''
        return client_loc
    
    def get_inspector_grading(self):
        string_list = self.get_string_list()
        grade_list = [s for s in string_list if self.checkmark_identifier in s]
        category_list = []
        inspected_list = []
        not_inspected_list = []
        not_present_list = []
        deficient_list = []
        
        for i, x in enumerate(grade_list):
            y = x.replace('⬜', '0').replace('☑', '1').replace('.', '').split(' ')
            y = [z for z in y if z != '']
            checkmarks = [int(n) for n in y[0:4]]
            category_list.append(' '.join(y[5:]))
            inspected_list.append(checkmarks[0])
            not_inspected_list.append(checkmarks[1])
            not_present_list.append(checkmarks[2])
            deficient_list.append(checkmarks[3])
        
        output_df = pd.DataFrame({'category' : category_list,
                                  'inspected' : inspected_list,
                                  'not_inspected' : not_inspected_list,
                                  'not_present' : not_present_list,
                                  'deficient' : deficient_list})
        return output_df
    
    def generate_table(self):
        grading = self.get_inspector_grading()
        grading['inspector_name'] = self.get_inspector_name()
        grading['inspection_date'] = self.get_inspection_date()
        grading['client_name'] = self.get_client_name()
        grading['client_location'] = self.get_client_location()
        return grading
    



### Initiate & Execute Class
###############################################################################
inspection_text = InspectionPdfReader(pdf_file_path = config_file_path)
table = inspection_text.generate_table()






























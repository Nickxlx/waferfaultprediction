import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{error}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message

# to check only
# cmd -> python src/exception.py
if __name__=="__main__":
    logging.info("Logging has started")

    try:
        a=1/0
    except Exception as e:
        logging.info('Division by zero') 
        raise CustomException(e,sys)
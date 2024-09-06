import sys
import logging
# error message detail, whenever an exception is raised this custom message will be pushed
def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Corrected line
    error_message = f"Error occurred in script: {file_name} at line number: {exc_tb.tb_lineno} error message: {str(error_message)}"
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)            
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
        


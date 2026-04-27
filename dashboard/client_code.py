from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Automatically link the button you created in the designer to the function below!
    # This fixes the issue where the button does nothing when clicked.
    if hasattr(self, 'run_button'):
        self.run_button.set_event_handler('click', self.run_button_click)

  def run_button_click(self, **event_args):
    """This method is called when the button is clicked"""
    
    # 1. Check if they actually created the components
    if not hasattr(self, 'file_loader'):
        alert("You need to create a FileLoader component named 'file_loader'")
        return
        
    # 2. Check if a file is uploaded
    if not self.file_loader.file:
        if hasattr(self, 'output_text'):
            self.output_text.text = "⚠️ Please click 'Upload Dataset' and select a CSV file first!"
        else:
            alert("Please upload a CSV file first.")
        return

    # Update status
    if hasattr(self, 'output_text'):
        self.output_text.text = "⏳ Running analysis... Please wait."

    try:
      # Call backend
      result = anvil.server.call('process_pipeline', self.file_loader.file)

      # Check for errors from backend
      if "error" in result:
        if hasattr(self, 'output_text'):
            self.output_text.text = result["error"]
        else:
            alert(result["error"])
        return

      # Display summary
      if hasattr(self, 'output_text'):
          self.output_text.text = result["summary"]

      # Display plot
      if hasattr(self, 'plot_1'):
          self.plot_1.source = result["plot"]

    except Exception as e:
      error_msg = f"An error occurred: {str(e)}"
      if hasattr(self, 'output_text'):
          self.output_text.text = error_msg
      else:
          alert(error_msg)

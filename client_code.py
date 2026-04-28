from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Link run_button click event
    if hasattr(self, 'run_button'):
      self.run_button.set_event_handler('click', self.run_button_click)

    # Link optional download button if present
    if hasattr(self, 'download_button'):
      self.download_button.set_event_handler('click', self.download_button_click)

  # ──────────────────────────────────────────────
  # HELPER: set status label with color coding
  # ──────────────────────────────────────────────
  def _set_status(self, bias_gap, threshold=0.10):
    """Display color-coded bias status if 'status_label' component exists."""
    if not hasattr(self, 'status_label'):
      return
    if bias_gap > threshold:
      self.status_label.text            = "⚠️ Bias Detected"
      self.status_label.foreground      = "#e53935"   # red
      self.status_label.bold            = True
    else:
      self.status_label.text            = "✅ Fair Model"
      self.status_label.foreground      = "#43a047"   # green
      self.status_label.bold            = True

  # ──────────────────────────────────────────────
  # HELPER: safe text setter
  # ──────────────────────────────────────────────
  def _set_text(self, component_name, text):
    comp = getattr(self, component_name, None)
    if comp is not None:
      comp.text = text

  # ──────────────────────────────────────────────
  # MAIN BUTTON: Run Analysis
  # ──────────────────────────────────────────────
  def run_button_click(self, **event_args):
    """Called when the Run Analysis button is clicked."""

    # ── Guard: FileLoader component must exist ──
    if not hasattr(self, 'file_loader'):
      alert("You need a FileLoader component named 'file_loader' in the designer.")
      return

    # ── Guard: file must be selected ──
    if not self.file_loader.file:
      self._set_text('output_text', "⚠️ Please click 'Upload Dataset' and select a CSV file first!")
      return

    # ── Show running status ──
    self._set_text('output_text', "⏳ Running analysis… Please wait.")

    # Clear previous plots and status
    if hasattr(self, 'plot_1'):
      self.plot_1.source = None
    if hasattr(self, 'plot_after'):
      self.plot_after.source = None
    if hasattr(self, 'status_label'):
      self.status_label.text = ""

    try:
      # ── Call backend pipeline ──
      result = anvil.server.call('process_pipeline', self.file_loader.file)

      # ── Handle backend errors ──
      if "error" in result:
        self._set_text('output_text', result["error"])
        return

      # ── Display summary text ──
      self._set_text('output_text', result.get("summary", ""))

      # ── Display BEFORE plot → plot_1 ──
      if hasattr(self, 'plot_1') and result.get("plot_before"):
        self.plot_1.source = result["plot_before"]

      # ── Display AFTER plot → plot_after ──
      if hasattr(self, 'plot_after') and result.get("plot_after"):
        self.plot_after.source = result["plot_after"]

      # ── Color-coded bias status ──
      bias_before = result.get("bias_before", 0)
      threshold = result.get("threshold", 0.10)
      self._set_status(bias_before, threshold)

    except Exception as e:
      error_msg = f"An error occurred: {str(e)}"
      self._set_text('output_text', error_msg)

  # ──────────────────────────────────────────────
  # DOWNLOAD BUTTON (placeholder)
  # ──────────────────────────────────────────────
  def download_button_click(self, **event_args):
    """Placeholder: future PDF/CSV report download."""
    alert("📄 Report download coming soon!\n\nThis feature will export a full bias analysis report.")

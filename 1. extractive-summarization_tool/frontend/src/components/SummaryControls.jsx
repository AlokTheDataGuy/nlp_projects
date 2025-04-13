import React from 'react';

const SummaryControls = ({ settings, onSettingsChange, onSubmit, disabled }) => {
  const handleChange = (field, value) => {
    onSettingsChange({
      ...settings,
      [field]: value
    });
  };

  return (
    <div className="summary-controls-section">
      <h2 className="section-title">Summary Settings</h2>
      
      <div className="control-group">
        <label htmlFor="ratio" className="control-label">
          Summary Length (% of original text)
          <span className="value-display">{Math.round(settings.ratio * 100)}%</span>
        </label>
        <input
          type="range"
          id="ratio"
          min="0.1"
          max="0.7"
          step="0.05"
          value={settings.ratio}
          onChange={(e) => handleChange('ratio', parseFloat(e.target.value))}
          className="range-slider"
        />
        <div className="range-labels">
          <span>10%</span>
          <span>70%</span>
        </div>
      </div>

      <div className="control-row">
        <div className="control-group half">
          <label htmlFor="min" className="control-label">
            Min. Sentences
          </label>
          <input
            type="number"
            id="min"
            min="1"
            max="10"
            value={settings.min}
            onChange={(e) => handleChange('min', parseInt(e.target.value, 10))}
            className="number-input"
          />
        </div>

        <div className="control-group half">
          <label htmlFor="max" className="control-label">
            Max. Sentences
          </label>
          <input
            type="number"
            id="max"
            min="5"
            max="50"
            value={settings.max}
            onChange={(e) => handleChange('max', parseInt(e.target.value, 10))}
            className="number-input"
          />
        </div>
      </div>

      <button 
        className="submit-button"
        onClick={onSubmit}
        disabled={disabled}
      >
        {disabled ? 'Upload a file first' : 'Generate Summary'}
      </button>
    </div>
  );
};

export default SummaryControls;
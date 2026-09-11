import { Calculator, CheckCircle2 } from 'lucide-react';

export default function CalculationView({ formula, givenValues, steps = [], result }) {
  return (
    <div className="vait-calc-card">
      <div className="vait-calc-header">
        <Calculator size={15} className="vait-calc-icon" />
        <span className="vait-calc-title">Step-by-Step Calculation</span>
      </div>

      {formula && (
        <div className="vait-calc-formula-box">
          <span className="calc-step-label">Formula:</span>
          <code>{formula}</code>
        </div>
      )}

      {givenValues && (
        <div className="vait-calc-given">
          <span className="calc-step-label">Given:</span>
          <span>{givenValues}</span>
        </div>
      )}

      {steps && steps.length > 0 && (
        <div className="vait-calc-steps">
          <span className="calc-step-label">Steps:</span>
          <ol>
            {steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      )}

      {result && (
        <div className="vait-calc-result">
          <CheckCircle2 size={16} className="vait-calc-result-icon" />
          <span className="vait-calc-result-label">Final Answer:</span>
          <span className="vait-calc-result-value">{result}</span>
        </div>
      )}
    </div>
  );
}

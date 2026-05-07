import React from 'react';

export interface ChoiceOption<T extends string> {
  id: T;
  label: React.ReactNode;
  description?: React.ReactNode;
  count?: React.ReactNode;
  tone?: 'danger';
  disabled?: boolean;
}

interface ChoiceGroupProps<T extends string> {
  options: ChoiceOption<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  columns?: 1 | 2 | 3;
  className?: string;
}

export function ChoiceGroup<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  columns = 2,
  className,
}: ChoiceGroupProps<T>) {
  return (
    <div
      className={`choice-group choice-group-${columns}${className ? ` ${className}` : ''}`}
      role="group"
      aria-label={ariaLabel}
    >
      {options.map((option) => {
        const active = value === option.id;
        return (
          <button
            key={option.id}
            type="button"
            className={`choice-btn${active ? ' active' : ''}${option.tone === 'danger' ? ' danger' : ''}`}
            onClick={() => onChange(option.id)}
            disabled={option.disabled}
            aria-pressed={active}
          >
            <span className="choice-btn-main">
              <span>{option.label}</span>
              {option.count && <span className="choice-btn-count">{option.count}</span>}
            </span>
            {option.description && <span className="choice-btn-description">{option.description}</span>}
          </button>
        );
      })}
    </div>
  );
}

import type { InfoContent } from '../types';

interface InfoModalProps {
  content: InfoContent;
  onClose: () => void;
}

/** Kurzes Info-Popup zur Vermittlung von Lerninhalten. */
export default function InfoModal({ content, onClose }: InfoModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-labelledby="info-title"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-start gap-3">
          <span className="text-2xl" aria-hidden="true">
            💡
          </span>
          <h3 id="info-title" className="text-lg font-bold text-brand-800">
            {content.title}
          </h3>
        </div>
        <p className="text-sm leading-relaxed text-slate-700">{content.body}</p>
        <button
          type="button"
          onClick={onClose}
          className="mt-5 w-full rounded-lg bg-brand-600 px-4 py-2.5 font-semibold text-white transition hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-400"
        >
          Verstanden
        </button>
      </div>
    </div>
  );
}

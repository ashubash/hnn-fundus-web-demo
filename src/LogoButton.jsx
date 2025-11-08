// src/LogoButton.jsx
import React from "react";
/**
 * LogoButton
 *
 * Props:
 * - isOpen?: boolean
 * When true (e.g. modal open), all animations are paused.
 * - onClick: () => void
 * Called when the logo is clicked or activated via keyboard.
 * - size?: number
 * Pixel size of the rendered SVG (width & height). Default: 140.
 * - title?: string
 * Tooltip + aria-label text. Default: "Model Details".
 */
export default function LogoButton({
  isOpen = false,
  onClick,
  size = 140,
  title = "Model Details",
}) {
  const handleKeyDown = (e) => {
    if (!onClick) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick();
    }
  };
  return (
    <>
      {/* Scoped styles for the logo button */}
      <style>{`
        /* Root wrapper just to allow easy inline-block placement */
        .logo-btn-root {
          display: inline-block;
        }
        .logo-btn-svg {
          display: block;
          cursor: pointer;
          outline: none;
          /* Isolate from the rest of layout/paint for performance */
          contain: layout style paint;
        }
        /* ===== Desktop-first animations ===== */
        .logo-btn-breathe-group {
          animation: logo-btn-breathe 8s ease-in-out infinite;
          transform-origin: center;
        }
        .logo-btn-pulse-ripple {
          transform-origin: center;
          animation: logo-btn-soft-glow-pulse 8s ease-out infinite;
          opacity: 0;
        }
        .logo-btn-pulse-ripple-delayed {
          animation-delay: 1.5s;
        }
        /* Pause all animations when flagged as open (e.g., modal visible) */
        .logo-btn-paused .logo-btn-breathe-group,
        .logo-btn-paused .logo-btn-pulse-ripple,
        .logo-btn-paused .logo-btn-pulse-ripple-delayed {
          animation-play-state: paused !important;
        }
        /* Hover / focus effects only on devices that support hover (desktops) */
        @media (hover: hover) and (pointer: fine) {
          .logo-btn-svg:hover,
          .logo-btn-svg:focus-visible {
            transform: translateY(-4px);
            transition: transform 0.35s ease;
          }
          .logo-btn-svg:hover .logo-btn-circle,
          .logo-btn-svg:focus-visible .logo-btn-circle {
            fill: #4A90E2 !important;
          }
        }
        /* Soft glow pulse – slightly optimized */
        @keyframes logo-btn-soft-glow-pulse {
          0%, 62.5% {
            transform: scale(0.05);
            opacity: 0;
          }
          68.125% {
            opacity: 0.3;
          }
          86.875% {
            opacity: 0.12;
          }
          100% {
            transform: scale(2.0); /* reduced from 3.0 to limit overdraw */
            opacity: 0;
          }
        }
        /* Breathe: subtle scale near the end of the cycle */
        @keyframes logo-btn-breathe {
          0%, 88.75% {
            transform: scale(1);
          }
          93.25% {
            transform: scale(1.06);
          }
          100% {
            transform: scale(1);
          }
        }
        /* ===== Respect prefers-reduced-motion ===== */
        @media (prefers-reduced-motion: reduce) {
          .logo-btn-svg,
          .logo-btn-breathe-group,
          .logo-btn-pulse-ripple,
          .logo-btn-pulse-ripple-delayed {
            animation: none !important;
            transition: none !important;
            transform: none !important;
          }
        }
        /* ===== Mobile-safe overrides ===== */
        @media (max-width: 600px) {
          /* Remove heavy clipping on mobile; let it render naturally */
          .logo-btn-svg {
            clip-path: none !important;
          }
          /* Disable pulse ripples & filters on mobile (expensive) */
          .logo-btn-pulse-ripple,
          .logo-btn-pulse-ripple-delayed {
            display: none !important;
            animation: none !important;
            filter: none !important;
            opacity: 0 !important;
          }
          /* Use a simple, cheap breathe animation */
          .logo-btn-breathe-group {
            animation: logo-btn-mobile-breathe 4s ease-in-out infinite;
          }
          @keyframes logo-btn-mobile-breathe {
            0%, 100% {
              transform: scale(1);
            }
            50% {
              transform: scale(1.13);
            }
          }
        }
      `}</style>
      <div className="logo-btn-root">
        <svg
          className={
            "logo-btn-svg" + (isOpen ? " logo-btn-paused" : "")
          }
          width={size}
          height={size}
          viewBox="0 0 40 40"
          xmlns="http://www.w3.org/2000/svg"
          onClick={onClick}
          onKeyDown={handleKeyDown}
          role="button"
          aria-label={title}
          tabIndex={0}
          title={title}
        >
          <defs>
            {/* Note: attribute names are camelCased for React/JSX */}
            <filter
              id="logo-btn-softglow"
              x="-200%"
              y="-200%"
              width="500%"
              height="500%"
            >
              <feGaussianBlur stdDeviation="5.5" result="blur" />
              <feFlood floodColor="#dcd4efff" floodOpacity="0.48" />
              <feComposite in2="blur" operator="in" />
              <feMerge>
                <feMergeNode />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <clipPath id="logo-btn-circle-clip">
              <circle cx="20" cy="20" r="12" />
            </clipPath>
          </defs>
          <g clipPath="url(#logo-btn-circle-clip)">
            <g className="logo-btn-breathe-group">
              {/* Main circle */}
              <circle
                className="logo-btn-circle"
                cx="20"
                cy="20"
                r="11.5"
                fill="#4A90E2"
                stroke="#ffffff"
                strokeWidth="1.3"
              />
              {/* Eyes */}
              <circle cx="16.2" cy="15" r="2" fill="#ffffff" />
              <circle cx="23.8" cy="15" r="2" fill="#ffffff" />
              {/* Smile path with MODEL text */}
              <path
                id="logo-btn-smilePath"
                d="M 10 21 Q 20 29 30 21"
                fill="none"
              />
              <text
                fontSize="2.45"
                fill="#ffffff"
                fontFamily="Arial, sans-serif"
                fontWeight="bold"
                textAnchor="middle"
              >
                <textPath
                  href="#logo-btn-smilePath"
                  startOffset="50%"
                  textLength="10.5"
                  dy="-5.8"
                >
                  MODEL
                </textPath>
              </text>
              {/* Pulsing ripples (desktop; hidden on mobile via CSS) */}
              <g
                className="logo-btn-pulse-ripple"
                filter="url(#logo-btn-softglow)"
              >
                <circle cx="20" cy="20" r="15" fill="#ffffff" />
              </g>
              <g
                className="logo-btn-pulse-ripple logo-btn-pulse-ripple-delayed"
                filter="url(#logo-btn-softglow)"
              >
                <circle cx="20" cy="20" r="15" fill="#ffffff" />
              </g>
            </g>
          </g>
        </svg>
      </div>
    </>
  );
}
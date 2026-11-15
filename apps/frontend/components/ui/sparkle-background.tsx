"use client";

import { useEffect, useRef } from "react";

export function SparkleBackground({ density = 25 }: { density?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const particles: HTMLDivElement[] = [];

    for (let i = 0; i < density; i++) {
      const particle = document.createElement("div");
      particle.className =
        "pointer-events-none fixed rounded-full bg-[#FFCA6E]";
      particle.style.width = `${1 + Math.random() * 2}px`;
      particle.style.height = particle.style.width;
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.zIndex = "0";
      particle.style.pointerEvents = "none";
      particle.style.animation = `sparkle-float ${
        8 + Math.random() * 12
      }s linear infinite`;
      particle.style.animationDelay = `${Math.random() * 10}s`;
      particle.style.opacity = "0";
      container.appendChild(particle);
      particles.push(particle);
    }

    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes sparkle-float {
        0% { transform: translateY(100vh) scale(0); opacity: 0; }
        10% { opacity: 0.9; }
        90% { opacity: 0.9; }
        100% { transform: translateY(-10vh) scale(1); opacity: 0; }
      }
    `;
    document.head.appendChild(style);

    return () => {
      particles.forEach((p) => p.remove());
      style.remove();
    };
  }, [density]);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 pointer-events-none z-0"
      aria-hidden="true"
    />
  );
}
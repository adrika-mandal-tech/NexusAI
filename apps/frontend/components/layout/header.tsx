"use client";

import { MobileSidebar } from "./mobile-sidebar";
import { NavbarSocialLinks } from "./contact-developer";

export function Header() {
  return (
    <header className="flex h-16 items-center justify-between border-b border-border/50 bg-background/80 backdrop-blur-xl px-4 sm:px-6 lg:px-8 sticky top-0 z-30 shadow-[0_0_30px_rgba(245,158,11,0.05)]">
      <div className="flex items-center gap-4">
        <MobileSidebar />
      </div>
      <div className="flex items-center gap-3 sm:gap-4">
        <span className="text-sm font-medium text-primary">NexusAI</span>
        <div className="hidden sm:block h-5 w-px bg-border/60"></div>
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            type="button"
            aria-label="Terminal"
            className="hidden sm:flex text-muted-foreground hover:text-primary p-2 rounded-full hover:bg-white/5 transition-all"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
            >
              <polyline points="4 17 10 11 4 5" />
              <line x1="12" x2="20" y1="19" y2="19" />
            </svg>
          </button>
          <NavbarSocialLinks />
        </div>
      </div>
    </header>
  );
}
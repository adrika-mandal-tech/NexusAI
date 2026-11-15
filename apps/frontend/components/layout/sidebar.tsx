"use client";

import Link from "next/link";
import { PrimaryNav } from "./primary-nav";

export function Sidebar() {
  return (
    <aside className="hidden md:flex w-56 flex-col h-screen border-r border-border bg-card/60 backdrop-blur-xl fixed left-0 top-0 z-40">
      <div className="flex h-16 items-center border-b border-border px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#F59E0B] to-[#CC5800] shadow-[0_0_20px_rgba(245,158,11,0.35)]">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4 text-white"
            >
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </div>
          <span className="bg-gradient-to-r from-[#FFCA6E] via-[#FFB347] to-[#CC5800] bg-clip-text text-transparent">
            NexusAI
          </span>
        </Link>
      </div>
      <div className="px-6 mb-8">
        <button className="w-full btn-primary-gradient py-3 rounded-lg font-bold shadow-[0_0_20px_rgba(245,158,11,0.25)] hover:shadow-[0_0_35px_rgba(245,158,11,0.45)] transition-all flex items-center justify-center gap-2 text-sm">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4"><line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/></svg>
          New Analysis
        </button>
      </div>
      <PrimaryNav />
    </aside>
  );
}
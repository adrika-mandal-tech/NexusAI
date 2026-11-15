"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database,
  Network,
  Bot,
  FileText,
  Activity,
  Globe,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ContactDeveloper } from "./contact-developer";

const mainNavItems = [
  {
    title: "Workspace",
    href: "/work",
    icon: Database,
  },
  {
    title: "AI Chat",
    href: "/chat",
    icon: Bot,
  },
  {
    title: "Knowledge Graph",
    href: "/graph",
    icon: Network,
  },
  {
    title: "Web & Retrieval",
    href: "/research",
    icon: Globe,
  },
  {
    title: "RAG Diagnostics & Eval",
    href: "/evaluation",
    icon: Activity,
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
  },
  {
    title: "Documentation",
    href: "/doc",
    icon: FileText,
  },
];

export function PrimaryNav() {
  const pathname = usePathname();

  return (
    <>
      <nav className="flex-1 space-y-1 p-4 overflow-y-auto">
        {mainNavItems.map((item) => {
          const isActive = pathname?.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all duration-200 scale-[0.98] active:scale-95",
                isActive
                  ? "bg-primary/15 text-primary font-medium border-r-2 border-primary"
                  : "text-muted-foreground hover:bg-white/5 hover:text-primary-fixed-dim"
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.title}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border p-3 space-y-2">
        <ContactDeveloper variant="sidebar" />
        <div className="pt-2 border-t border-border/50">
          <Link
            href="/"
            className={cn(
              "flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-all duration-200 scale-[0.98] active:scale-95",
              pathname === "/"
                ? "bg-primary/15 text-primary font-medium"
                : "text-muted-foreground hover:bg-white/5 hover:text-primary-fixed-dim"
            )}
          >
            <span className="text-xs font-medium">Home</span>
          </Link>
        </div>
      </div>
    </>
  );
}

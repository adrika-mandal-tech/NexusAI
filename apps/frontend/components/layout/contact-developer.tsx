"use client";

import { useState } from "react";
import { Github } from "lucide-react";
import { cn } from "@/lib/utils";

const DEVELOPER_EMAIL = "adrika.mandal01@gmail.com";
const DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/adrika-mandal-753226246/";
const REPO_GITHUB = "https://github.com/adrika-mandal-tech/NexusAI";

export function GmailLogo({
  className,
  active = false,
}: {
  className?: string;
  active?: boolean;
}) {
  if (active) {
    return (
      <svg
        viewBox="0 0 24 24"
        className={cn("h-4 w-4 shrink-0", className)}
        aria-hidden="true"
      >
        <path
          fill="#FACC15"
          d="M20 5H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2v-6.5l6 4.5 6-4.5V19h2a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm-8 9.25L5.2 9.15A1.5 1.5 0 0 1 6 6.75L12 11.25l6-4.5a1.5 1.5 0 0 1 .8 2.4L12 14.25Z"
        />
      </svg>
    );
  }

  return (
    <svg
      viewBox="0 0 24 24"
      className={cn("h-4 w-4 shrink-0", className)}
      aria-hidden="true"
    >
      <path
        fill="#4285F4"
        d="M2 9.5L6 12.5V19H3.5A1.5 1.5 0 0 1 2 17.5V9.5Z"
      />
      <path
        fill="#34A853"
        d="M22 9.5L18 12.5V19H20.5A1.5 1.5 0 0 0 22 17.5V9.5Z"
      />
      <path
        fill="#EA4335"
        d="M6 12.5V6.2L12 10.7L18 6.2V12.5L12 17L6 12.5Z"
      />
      <path
        fill="#C5221F"
        d="M2 9.5V6.8A1.8 1.8 0 0 1 4.9 5.35L6 6.2V12.5L2 9.5Z"
      />
      <path
        fill="#FBBC05"
        d="M22 9.5V6.8A1.8 1.8 0 0 0 19.1 5.35L18 6.2V12.5L22 9.5Z"
      />
    </svg>
  );
}

export function LinkedInLogo({
  className,
  active = false,
}: {
  className?: string;
  active?: boolean;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={cn("h-4 w-4 shrink-0", className)}
      aria-hidden="true"
    >
      <rect
        width="24"
        height="24"
        rx="4"
        fill={active ? "#FACC15" : "#0A66C2"}
      />
      <path
        fill={active ? "#09090B" : "#FFFFFF"}
        d="M7.5 9.5H5v9h2.5v-9ZM6.25 8.4a1.45 1.45 0 1 0 0-2.9 1.45 1.45 0 0 0 0 2.9ZM19 13.3c0-2.55-1.35-3.95-3.4-3.95-1.45 0-2.25.78-2.6 1.38V9.5h-2.4v9H13v-4.8c0-1.26.62-2.05 1.75-2.05 1.05 0 1.65.72 1.65 2.05V18.5H19v-5.2Z"
      />
    </svg>
  );
}

const CONTACT_LINKS = [
  {
    id: "gmail",
    label: "Gmail",
    href: `mailto:${DEVELOPER_EMAIL}`,
    title: "Gmail",
    external: false,
    Logo: GmailLogo,
  },
  {
    id: "linkedin",
    label: "LinkedIn",
    href: DEVELOPER_LINKEDIN,
    title: "LinkedIn",
    external: true,
    Logo: LinkedInLogo,
  },
] as const;

export function NavbarSocialLinks() {
  const [clickedId, setClickedId] = useState<string | null>(null);

  const items = [
    {
      id: "github",
      href: REPO_GITHUB,
      tooltip: "View NexusAI on GitHub",
      external: true,
      renderIcon: (active: boolean) => (
        <Github
          className={cn(
            "h-4 w-4 transition-transform duration-200 group-hover:scale-110",
            active ? "text-yellow-400" : ""
          )}
        />
      ),
    },
    {
      id: "linkedin",
      href: DEVELOPER_LINKEDIN,
      tooltip: "LinkedIn",
      external: true,
      renderIcon: (active: boolean) => (
        <LinkedInLogo
          active={active}
          className="h-3.5 w-3.5 transition-transform duration-200 group-hover:scale-110"
        />
      ),
    },
    {
      id: "gmail",
      href: `mailto:${DEVELOPER_EMAIL}`,
      tooltip: "Gmail",
      external: false,
      renderIcon: (active: boolean) => (
        <GmailLogo
          active={active}
          className="h-3.5 w-3.5 transition-transform duration-200 group-hover:scale-110"
        />
      ),
    },
  ];

  return (
    <div className="flex items-center gap-1.5 sm:gap-2">
      {items.map(({ id, href, tooltip, external, renderIcon }) => {
        const isActive = clickedId === id;
        return (
          <a
            key={id}
            href={href}
            {...(external
              ? { target: "_blank", rel: "noopener noreferrer" }
              : {})}
            title={tooltip}
            aria-label={tooltip}
            onClick={() => setClickedId(id)}
            className={cn(
              "group relative flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-full border transition-all duration-200",
              "active:border-yellow-400 active:bg-yellow-400/25 active:text-yellow-400 active:scale-95",
              isActive
                ? "border-yellow-400 bg-yellow-400/20 text-yellow-400 shadow-[0_0_12px_rgba(250,204,21,0.5)]"
                : "border-border/50 text-muted-foreground hover:text-yellow-400 hover:bg-yellow-400/10 hover:border-yellow-400/50"
            )}
          >
            {renderIcon(isActive)}
            <span className="sr-only">{tooltip}</span>
            <span className="pointer-events-none absolute -bottom-9 right-0 z-50 hidden rounded-md bg-popover px-2.5 py-1 text-[11px] font-medium text-popover-foreground shadow-lg border border-border/60 whitespace-nowrap group-hover:block transition-all">
              {tooltip}
            </span>
          </a>
        );
      })}
    </div>
  );
}

export function ContactDeveloper({
  variant = "sidebar",
}: {
  variant?: "sidebar" | "footer";
}) {
  const [clickedId, setClickedId] = useState<string | null>(null);

  if (variant === "footer") {
    return (
      <div className="pt-5 border-t border-border/40 flex flex-col sm:flex-row items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Contact Developer
        </span>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {CONTACT_LINKS.map(({ id, label, href, title, external, Logo }) => {
            const isActive = clickedId === id;
            return (
              <a
                key={id}
                href={href}
                title={title}
                aria-label={title}
                onClick={() => setClickedId(id)}
                {...(external
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                className={cn(
                  "flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs transition-all duration-200",
                  "active:border-yellow-400 active:bg-yellow-400/20 active:text-yellow-400",
                  isActive
                    ? "border-yellow-400 bg-yellow-400/20 text-yellow-400 shadow-[0_0_12px_rgba(250,204,21,0.45)]"
                    : "border-border/60 bg-card/60 text-foreground/90 hover:border-yellow-400/50 hover:bg-yellow-400/10 hover:text-yellow-400"
                )}
              >
                <Logo active={isActive} className="h-4 w-4" />
                <span>{label}</span>
              </a>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1.5 pb-2">
      <p className="px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80">
        Contact Developer
      </p>
      <div className="flex flex-col gap-1">
        {CONTACT_LINKS.map(({ id, label, href, title, external, Logo }) => {
          const isActive = clickedId === id;
          return (
            <a
              key={id}
              href={href}
              title={title}
              aria-label={title}
              onClick={() => setClickedId(id)}
              {...(external
                ? { target: "_blank", rel: "noopener noreferrer" }
                : {})}
              className={cn(
                "flex items-center gap-2.5 rounded-lg border px-2.5 py-1.5 text-xs transition-all duration-200 group",
                "active:border-yellow-400 active:bg-yellow-400/20 active:text-yellow-400",
                isActive
                  ? "border-yellow-400/80 bg-yellow-400/15 text-yellow-400 font-medium"
                  : "border-transparent text-muted-foreground hover:bg-white/5 hover:text-yellow-400"
              )}
            >
              <Logo
                active={isActive}
                className="h-3.5 w-3.5 transition-transform duration-200 group-hover:scale-110"
              />
              <span className="truncate">{label}</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}

"use client";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu } from "lucide-react";
import { PrimaryNav } from "./primary-nav";

export function MobileSidebar() {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className="md:hidden border-border/60">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle navigation menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="p-0 flex flex-col bg-card/95 backdrop-blur-xl">
        <div className="flex h-16 items-center border-b border-border px-6">
          <span className="font-headline-md text-headline-md font-bold text-primary">NexusAI</span>
        </div>
        <PrimaryNav />
      </SheetContent>
    </Sheet>
  );
}
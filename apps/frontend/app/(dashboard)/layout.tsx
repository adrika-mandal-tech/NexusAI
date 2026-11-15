import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { SparkleBackground } from "@/components/ui/sparkle-background";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen bg-background overflow-hidden relative">
      <SparkleBackground density={18} />
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden relative z-10 md:ml-56">
        <Header />
        <main className="flex-1 overflow-y-auto bg-background/50 p-4 sm:p-6 lg:p-8">
          <div className="mx-auto max-w-6xl w-full h-full animate-fade-in">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router";
import { Loader2 } from "lucide-react";

import { AppHeader } from "@/components/layout/AppHeader";
import CreateWorkflow from "@/pages/CreateWorkflow";
import RunHistory from "@/pages/RunHistory";

// Split out: React Flow is a large dependency and only this route needs it.
// Loading it on the create page would make the first paint pay for a graph the
// visitor has not asked for yet.
const WorkflowExecution = lazy(() => import("@/pages/WorkflowExecution"));

export default function App() {
  return (
    <div className="min-h-screen">
      <AppHeader />
      <main>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<CreateWorkflow />} />
            <Route path="/history" element={<RunHistory />} />
            {/* The run id is in the URL so a run survives a refresh and can be
                shared - the backend retains finished runs for a while. */}
            <Route path="/runs/:runId" element={<WorkflowExecution />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}

function RouteFallback() {
  return (
    <div className="flex items-center justify-center py-24 text-muted-foreground">
      <Loader2 className="size-5 animate-spin" aria-hidden="true" />
    </div>
  );
}

function NotFound() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-20 text-center">
      <h1 className="font-heading text-2xl font-semibold">Page not found</h1>
      <p className="mt-2 text-muted-foreground">
        Nothing lives at this address. Head back to start a workflow.
      </p>
    </div>
  );
}

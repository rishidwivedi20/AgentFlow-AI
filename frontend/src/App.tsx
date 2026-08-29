import { Route, Routes } from "react-router";

import { AppHeader } from "@/components/layout/AppHeader";
import CreateWorkflow from "@/pages/CreateWorkflow";
import WorkflowExecution from "@/pages/WorkflowExecution";

export default function App() {
  return (
    <div className="min-h-screen">
      <AppHeader />
      <main>
        <Routes>
          <Route path="/" element={<CreateWorkflow />} />
          {/* The run id is in the URL so a run survives a refresh and can be
              shared - the backend retains finished runs for a while. */}
          <Route path="/runs/:runId" element={<WorkflowExecution />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
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

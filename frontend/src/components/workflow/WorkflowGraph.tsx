import { useMemo } from "react";
import { Background, MarkerType, ReactFlow } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";

import "@xyflow/react/dist/style.css";

import { AgentNode } from "@/components/workflow/AgentNode";
import type { AgentNodeData } from "@/components/workflow/AgentNode";
import { AGENT_NODES, WORKFLOW_EDGES } from "@/lib/workflow-graph";
import type { AgentProgress } from "@/lib/workflow-graph";
import type { AgentName } from "@/types/workflow";

// Defined outside the component: React Flow warns when this object identity
// changes between renders, which it would on every event if inlined.
const NODE_TYPES = { agent: AgentNode };

interface WorkflowGraphProps {
  progress: Record<AgentName, AgentProgress>;
}

/**
 * The workflow as a graph, with each node reflecting live execution state.
 *
 * Read-only: this visualises a run, it is not an editor. Interaction is
 * limited to panning and zooming so a node cannot be dragged out of a layout
 * that carries meaning.
 */
export function WorkflowGraph({ progress }: WorkflowGraphProps) {
  const nodes = useMemo<Node<AgentNodeData>[]>(
    () =>
      AGENT_NODES.map((spec) => ({
        id: spec.id,
        type: "agent",
        position: spec.position,
        data: { label: spec.label, role: spec.role, progress: progress[spec.id] },
        draggable: false,
      })),
    [progress],
  );

  const edges = useMemo<Edge[]>(
    () =>
      WORKFLOW_EDGES.map((spec) => {
        // A loop edge is "live" once its target has run more than once, which
        // is the only evidence available that the branch was actually taken.
        const taken = spec.isLoop && progress[spec.target].executions > 1;
        return {
          id: spec.id,
          source: spec.source,
          target: spec.target,
          sourceHandle: spec.isLoop ? "loop" : undefined,
          targetHandle: spec.isLoop ? "loop" : undefined,
          type: "smoothstep",
          animated: Boolean(taken),
          label: spec.label,
          labelStyle: { fontSize: 10, fill: "var(--muted-foreground)" },
          labelBgStyle: { fill: "var(--background)" },
          style: {
            stroke: taken ? "var(--color-amber-500)" : "var(--border)",
            strokeWidth: taken ? 2 : 1.5,
            strokeDasharray: spec.isLoop ? "4 3" : undefined,
          },
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
        };
      }),
    [progress],
  );

  return (
    <div className="h-[640px] w-full rounded-lg border bg-surface">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        proOptions={{ hideAttribution: false }}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        panOnScroll
        minZoom={0.4}
        maxZoom={1.5}
      >
        <Background gap={16} size={1} />
      </ReactFlow>
    </div>
  );
}

"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import Link from "next/link";
import {
  Network,
  Database,
  Search,
  Zap,
  Play,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Terminal,
  Code,
  Table as TableIcon,
  Layers,
  Sparkles,
  Upload,
  FileSpreadsheet,
  X,
  CheckCircle2,
  AlertCircle,
  Filter,
  ArrowRight,
  RefreshCw,
  FileUp,
  Trash2,
  Compass,
  Info,
  Activity,
  Sliders,
  ChevronRight,
  Maximize2,
  Tag,
  Share2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { apiFetch, executeCypher, CypherConsoleResult } from "@/lib/api-client";
import { authFetch } from "@/lib/auth";

export type NodeData = {
  id: string;
  label: string;
  x: number; // pixel coords in SVG canvas
  y: number; // pixel coords in SVG canvas
  vx?: number;
  vy?: number;
  r: number;
  desc?: string;
  type: string;
  labels?: string[];
  doc_type?: string | null;
  properties?: Record<string, any>;
  in_degree?: number;
  out_degree?: number;
  degree?: number;
  source_file?: string;
  orbitPhase?: number;
};

export type EdgeData = {
  id: string;
  source: string;
  target: string;
  type: string;
  label?: string;
  properties?: Record<string, any>;
  source_file?: string;
};

interface IngestedFileMeta {
  filename: string;
  node_count: number;
  sample_labels: string[];
}

// Authentic Neo4j Browser palette strictly for dataset entities
const NEO4J_PALETTE: Record<string, { bg: string; border: string; text: string; role: string }> = {
  Customer: { bg: "#C62828", border: "#E53935", text: "#FFFFFF", role: "Primary Client Node" },
  Company: { bg: "#7B1FA2", border: "#9C27B0", text: "#FFFFFF", role: "Corporate Entity Node" },
  City: { bg: "#00838F", border: "#00ACC1", text: "#FFFFFF", role: "Geographic City Node" },
  Country: { bg: "#2E7D32", border: "#43A047", text: "#FFFFFF", role: "Sovereign Country Node" },
  Order: { bg: "#E65100", border: "#FB8C00", text: "#FFFFFF", role: "Commercial Transaction Node" },
  Item: { bg: "#AD1457", border: "#D81B60", text: "#FFFFFF", role: "Product Item Node" },
  Product: { bg: "#1B5E20", border: "#2E7D32", text: "#FFFFFF", role: "Product Dimension Node" },
  Supplier: { bg: "#4527A0", border: "#5E35B1", text: "#FFFFFF", role: "Vendor Supplier Node" },
  ServiceType: { bg: "#00695C", border: "#00897B", text: "#FFFFFF", role: "Order Fulfillment Node" },
  Organization: { bg: "#C62828", border: "#E53935", text: "#FFFFFF", role: "Parent Organization Node" },
};

const RELATIONSHIP_TYPES = [
  { name: "LOCATED_IN", source: "Customer", target: "City", color: "#00BCD4" },
  { name: "IN_COUNTRY", source: "City", target: "Country", color: "#4CAF50" },
  { name: "AFFILIATED_WITH", source: "Customer", target: "Company", color: "#BA68C8" },
  { name: "PLACED", source: "Customer", target: "Order", color: "#FF9800" },
  { name: "CONTAINS", source: "Order", target: "Item", color: "#F06292" },
];

const DATASET_PROPERTIES = [
  "customerID",
  "companyName",
  "contactName",
  "city",
  "country",
  "postalCode",
  "order_id",
  "item_id",
  "quantity",
  "in_or_out",
];

function getNodeColor(type: string) {
  const match = NEO4J_PALETTE[type];
  if (match) return match;
  // Deterministic fallback based on string hash
  const keys = Object.keys(NEO4J_PALETTE);
  let hash = 0;
  for (let i = 0; i < type.length; i++) {
    hash = (hash << 5) - hash + type.charCodeAt(i);
  }
  const picked = keys[Math.abs(hash) % keys.length];
  return NEO4J_PALETTE[picked];
}

// Break labels for clean rendering INSIDE circles (just like Neo4j Browser)
function formatInnerLabel(text: string, radius: number): { line1: string; line2?: string } {
  if (!text) return { line1: "" };
  const clean = text.trim();

  // Very small leaf circle
  if (radius <= 17) {
    return { line1: clean.length > 7 ? `${clean.slice(0, 6)}…` : clean };
  }

  // Short word fits in one line
  if (clean.length <= 8) {
    return { line1: clean };
  }

  // Split multi-word labels
  const words = clean.split(/[\s_-]+/);
  if (words.length >= 2) {
    const l1 = words[0];
    const l2 = words.slice(1).join(" ");
    return {
      line1: l1.length > 9 ? `${l1.slice(0, 8)}…` : l1,
      line2: l2.length > 10 ? `${l2.slice(0, 9)}…` : l2,
    };
  }

  // Hyphenate middle of long word
  const mid = Math.ceil(clean.length / 2);
  const p1 = clean.slice(0, Math.min(mid, 8));
  const p2 = clean.slice(p1.length);
  return {
    line1: `${p1}-`,
    line2: p2.length > 9 ? `${p2.slice(0, 8)}…` : p2,
  };
}

const CANVAS_WIDTH = 1400;
const CANVAS_HEIGHT = 850;

export default function GraphPage() {
  const [activeTab, setActiveTab] = useState<"graph" | "table" | "raw">("graph");
  const [selected, setSelected] = useState<NodeData | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [filterLabel, setFilterLabel] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [motionActive, setMotionActive] = useState(true);

  // Ingested files & active dataset switcher
  const [ingestedFiles, setIngestedFiles] = useState<IngestedFileMeta[]>([]);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  // CSV/XLSX Upload Placeholder modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showPropertiesModal, setShowPropertiesModal] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadProcessing, setUploadProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Pan and zoom states
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0 });

  // Dragging individual node
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Cypher Console state - Closed by default & shortened height
  const [cypherOpen, setCypherOpen] = useState(false);
  const [cypherInput, setCypherInput] = useState("MATCH (c:Customer)-[r:LOCATED_IN]->(city:City) RETURN c, r, city LIMIT 35");
  const [cypherRunning, setCypherRunning] = useState(false);
  const [cypherResult, setCypherResult] = useState<CypherConsoleResult | null>(null);
  const [cypherError, setCypherError] = useState<string | null>(null);

  // Animation & Motion state references
  const nodesRef = useRef<NodeData[]>([]);
  const edgesRef = useRef<EdgeData[]>([]);
  const [, setRenderTrigger] = useState(0);
  const animFrameRef = useRef<number | null>(null);
  const [animationTick, setAnimationTick] = useState(0);

  // Radial Cluster Layout Initializer (Creates organic starwheel clusters)
  const initPositions = useCallback((rawNodes: any[], rawEdges: any[] = []): NodeData[] => {
    if (!rawNodes || rawNodes.length === 0) return [];

    const byType = new Map<string, any[]>();
    rawNodes.forEach((n) => {
      const type = n.type || (n.labels && n.labels[0]) || "Entity";
      if (!byType.has(type)) byType.set(type, []);
      byType.get(type)!.push(n);
    });

    const categories = Array.from(byType.keys()).sort();
    const totalCategories = Math.max(1, categories.length);

    const degreeMap: Record<string, { in: number; out: number }> = {};
    rawNodes.forEach((n) => {
      degreeMap[n.id] = { in: 0, out: 0 };
    });
    rawEdges.forEach((e) => {
      if (degreeMap[e.source]) degreeMap[e.source].out++;
      if (degreeMap[e.target]) degreeMap[e.target].in++;
    });

    const placed: NodeData[] = [];
    const centerX = CANVAS_WIDTH / 2;
    const centerY = CANVAS_HEIGHT / 2;

    categories.forEach((cat, catIdx) => {
      const group = byType.get(cat)!;
      const catAngle = (catIdx / totalCategories) * Math.PI * 2;
      const clusterDist = totalCategories > 1 ? 260 : 0;
      const clusterCenterX = centerX + Math.cos(catAngle) * clusterDist;
      const clusterCenterY = centerY + Math.sin(catAngle) * clusterDist;

      group.forEach((node, nodeIdx) => {
        const nodeAngle = (nodeIdx / Math.max(1, group.length)) * Math.PI * 2;
        const spreadRadius = group.length > 1 ? 95 + (nodeIdx % 4) * 28 : 0;
        const x = clusterCenterX + Math.cos(nodeAngle) * spreadRadius;
        const y = clusterCenterY + Math.sin(nodeAngle) * spreadRadius;

        const inDeg = degreeMap[node.id]?.in ?? (node.in_degree || 0);
        const outDeg = degreeMap[node.id]?.out ?? (node.out_degree || 0);
        const totalDeg = inDeg + outDeg;

        // Hub nodes get larger radius (28-36), leaves get compact radius (16-24)
        const isHub = totalDeg >= 4 || group.length >= 6;
        const r = node.r || (isHub ? Math.min(36, Math.max(26, 26 + totalDeg * 1.1)) : Math.min(22, Math.max(16, 16 + totalDeg * 0.9)));

        placed.push({
          ...node,
          id: String(node.id || node.label),
          label: String(node.label || node.properties?.name || node.id || "Node"),
          type: cat,
          x: Math.max(60, Math.min(CANVAS_WIDTH - 60, x)),
          y: Math.max(60, Math.min(CANVAS_HEIGHT - 60, y)),
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          r,
          in_degree: inDeg,
          out_degree: outDeg,
          degree: totalDeg,
          properties: node.properties || {},
          source_file: node.source_file,
          orbitPhase: Math.random() * Math.PI * 2,
        });
      });
    });

    return placed;
  }, []);

  const setGraphPayload = useCallback((nodes: NodeData[], edges: EdgeData[]) => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
    setRenderTrigger((v) => v + 1);
  }, []);

  // Continuous Dynamic Motion Loop (Gently moved, rotated, and observed with harmonic drift)
  useEffect(() => {
    let lastTime = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(32, now - lastTime);
      lastTime = now;
      const timeSec = now * 0.001;

      const nodes = nodesRef.current;
      const edges = edgesRef.current;

      if (motionActive && nodes.length > 0) {
        const centerX = CANVAS_WIDTH / 2;
        const centerY = CANVAS_HEIGHT / 2;
        const nodeMap = new Map<string, NodeData>();
        nodes.forEach((n) => nodeMap.set(n.id, n));

        // 1. Anti-collision repulsion
        const nLen = nodes.length;
        for (let i = 0; i < nLen; i++) {
          const n1 = nodes[i];
          for (let j = i + 1; j < nLen; j++) {
            const n2 = nodes[j];
            const dx = n2.x - n1.x;
            const dy = n2.y - n1.y;
            const distSq = dx * dx + dy * dy + 1;
            const minDist = n1.r + n2.r + 32;

            if (distSq < minDist * minDist * 6) {
              const dist = Math.sqrt(distSq);
              const force = (minDist * minDist * 35) / distSq;
              const fx = (dx / dist) * force;
              const fy = (dy / dist) * force;

              if (n1.id !== draggingNodeId) {
                n1.vx = (n1.vx || 0) - fx * 0.03;
                n1.vy = (n1.vy || 0) - fy * 0.03;
              }
              if (n2.id !== draggingNodeId) {
                n2.vx = (n2.vx || 0) + fx * 0.03;
                n2.vy = (n2.vy || 0) + fy * 0.03;
              }
            }
          }
        }

        // 2. Spring attraction along relationships
        for (const edge of edges) {
          const src = nodeMap.get(edge.source);
          const tgt = nodeMap.get(edge.target);
          if (src && tgt) {
            const dx = tgt.x - src.x;
            const dy = tgt.y - src.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const restLen = 110;
            const delta = dist - restLen;
            const springForce = delta * 0.015;

            const fx = (dx / dist) * springForce;
            const fy = (dy / dist) * springForce;

            if (src.id !== draggingNodeId) {
              src.vx = (src.vx || 0) + fx;
              src.vy = (src.vy || 0) + fy;
            }
            if (tgt.id !== draggingNodeId) {
              tgt.vx = (tgt.vx || 0) - fx;
              tgt.vy = (tgt.vy || 0) - fy;
            }
          }
        }

        // 3. Gentle harmonic breathing & center tether
        nodes.forEach((node) => {
          if (node.id === draggingNodeId) return;

          const dxCenter = centerX - node.x;
          const dyCenter = centerY - node.y;
          node.vx = (node.vx || 0) + dxCenter * 0.00035;
          node.vy = (node.vy || 0) + dyCenter * 0.00035;

          const phase = (node.orbitPhase || 0) + timeSec * 0.8;
          node.vx += Math.sin(phase) * 0.04;
          node.vy += Math.cos(phase) * 0.04;

          node.vx *= 0.88;
          node.vy *= 0.88;

          node.x += node.vx;
          node.y += node.vy;

          const margin = node.r + 20;
          node.x = Math.max(margin, Math.min(CANVAS_WIDTH - margin, node.x));
          node.y = Math.max(margin, Math.min(CANVAS_HEIGHT - margin, node.y));
        });

        setAnimationTick((t) => (t + 1) % 1000000);
      }

      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [motionActive, draggingNodeId]);

  // Load uploaded spreadsheet files list
  const loadIngestedFiles = useCallback(async () => {
    try {
      const files = await authFetch("/api/v1/graph/csv/files");
      if (Array.isArray(files)) {
        setIngestedFiles(files);
        if (files.length > 0 && !activeFile) {
          setActiveFile(files[0].filename);
        }
      }
    } catch (e) {
      console.warn("Could not load ingested files list", e);
    }
  }, [activeFile]);

  // Fetch knowledge graph from backend strictly for uploaded spreadsheet datasets
  const loadGraph = useCallback(async (targetFile?: string | null) => {
    setLoading(true);
    setStatusMessage("");
    try {
      const fileParam = targetFile !== undefined ? targetFile : activeFile;
      const url = fileParam
        ? `/api/v1/graph/csv/viz?source_file=${encodeURIComponent(fileParam)}&node_limit=250&edge_limit=500`
        : "/api/v1/graph/csv/viz?node_limit=250&edge_limit=500";

      const vizData = await authFetch(url);
      if (vizData && vizData.nodes && vizData.nodes.length > 0) {
        const laidOut = initPositions(vizData.nodes, vizData.edges || []);
        setGraphPayload(
          laidOut,
          (vizData.edges || []).map((e: any, idx: number) => ({
            id: e.id || `edge-${idx}`,
            source: String(e.source),
            target: String(e.target),
            type: e.label || e.type || "RELATED",
            source_file: e.source_file,
            properties: e.properties || {},
          }))
        );
        setStatusMessage(
          `Loaded ${vizData.nodes.length} nodes & ${vizData.edges?.length || 0} relationships from ${fileParam || "dataset"}.`
        );
        return;
      }

      // If no nodes found for specific file, try general CSV viz
      if (fileParam) {
        const generalData = await authFetch("/api/v1/graph/csv/viz?node_limit=250&edge_limit=500");
        if (generalData && generalData.nodes && generalData.nodes.length > 0) {
          const laidOut = initPositions(generalData.nodes, generalData.edges || []);
          setGraphPayload(
            laidOut,
            (generalData.edges || []).map((e: any, idx: number) => ({
              id: e.id || `edge-${idx}`,
              source: String(e.source),
              target: String(e.target),
              type: e.label || e.type || "RELATED",
              source_file: e.source_file,
              properties: e.properties || {},
            }))
          );
          setStatusMessage(`Loaded ${generalData.nodes.length} nodes & ${generalData.edges?.length || 0} relationships.`);
          return;
        }
      }

      // No dataset uploaded yet
      setGraphPayload([], []);
      setStatusMessage("No spreadsheet dataset ingested yet. Upload an Excel or CSV file to build the Knowledge Graph.");
    } catch (e) {
      console.error("Error loading dataset knowledge graph", e);
      setGraphPayload([], []);
      setStatusMessage("Error loading knowledge graph from database.");
    } finally {
      setLoading(false);
      loadIngestedFiles();
    }
  }, [activeFile, initPositions, setGraphPayload, loadIngestedFiles]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Delete spreadsheet dataset
  const handleDeleteFile = async (filename: string) => {
    if (!confirm(`Are you sure you want to delete '${filename}' and remove all its entity nodes from the knowledge graph?`)) {
      return;
    }
    setDeletingFile(filename);
    try {
      await authFetch(`/api/v1/graph/csv/files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      setStatusMessage(`Deleted '${filename}' and removed all associated nodes from Neo4j.`);
      if (activeFile === filename) {
        setActiveFile(null);
      }
      await loadGraph(null);
      await loadIngestedFiles();
    } catch (err: any) {
      alert(`Error deleting spreadsheet: ${err.message}`);
    } finally {
      setDeletingFile(null);
    }
  };

  // Run Cypher query (strictly against the dataset)
  const handleRunCypher = async (queryOverride?: string) => {
    const q = (queryOverride || cypherInput).trim();
    if (!q) return;
    setCypherRunning(true);
    setCypherError(null);
    try {
      const res = await executeCypher(q);
      setCypherResult(res);

      if (res.nodes && res.nodes.length > 0) {
        const laidOut = initPositions(res.nodes, res.edges || []);
        setGraphPayload(
          laidOut,
          (res.edges || []).map((e: any, idx: number) => ({
            id: e.id || `cypher-edge-${idx}`,
            source: e.source,
            target: e.target,
            type: e.type || e.label || "RELATED",
            properties: e.properties || {},
          }))
        );
        setStatusMessage(`Cypher returned ${res.nodes.length} nodes & ${res.edges?.length || 0} edges in ${res.elapsed_ms.toFixed(1)}ms.`);
      } else {
        setStatusMessage(`Query executed: ${res.row_count} rows returned.`);
      }
    } catch (err: any) {
      setCypherError(err.message || "Failed to execute Cypher query");
    } finally {
      setCypherRunning(false);
    }
  };

  // File Upload (Drag and Drop / File selector)
  const handleFileUpload = async () => {
    if (uploadFiles.length === 0) return;
    setUploadProcessing(true);
    setUploadError(null);
    setUploadSuccess(null);
    setUploadProgress("Analyzing spreadsheet structure and inferring entity schema...");

    try {
      const form = new FormData();
      uploadFiles.forEach((file) => form.append("files", file));

      const uploadRes = await authFetch("/api/v1/graph/csv/upload", {
        method: "POST",
        body: form,
      });

      const proposalId = uploadRes.proposal_id;
      if (proposalId) {
        await authFetch(`/api/v1/graph/csv/proposal/${proposalId}/approve`, {
          method: "POST",
          body: JSON.stringify({ review_comments: "Auto-approved from Graph Studio" }),
        });

        setUploadProgress("Generating Neo4j nodes and directional relationship vectors...");
        await authFetch("/api/v1/graph/csv/ingest", {
          method: "POST",
          body: JSON.stringify({ proposal_id: proposalId, dry_run: false }),
        });
      }

      setUploadSuccess(`Successfully ingested ${uploadFiles.length} file(s)! Knowledge Graph generated.`);
      setUploadProgress(null);
      setShowUploadModal(false);
      setUploadFiles([]);

      await loadIngestedFiles();
      await loadGraph(uploadFiles[0]?.name);
    } catch (err: any) {
      console.error("Upload error:", err);
      setUploadError(err.message || "Failed to process spreadsheet");
    } finally {
      setUploadProcessing(false);
    }
  };

  // Zoom handlers
  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.25, 3.5));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.25, 0.35));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsPanning(true);
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggingNodeId && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      const rawX = (e.clientX - rect.left - pan.x) / zoom;
      const rawY = (e.clientY - rect.top - pan.y) / zoom;

      const node = nodesRef.current.find((n) => n.id === draggingNodeId);
      if (node) {
        node.x = Math.max(30, Math.min(CANVAS_WIDTH - 30, rawX));
        node.y = Math.max(30, Math.min(CANVAS_HEIGHT - 30, rawY));
        node.vx = 0;
        node.vy = 0;
      }
      return;
    }

    if (isPanning) {
      setPan({
        x: e.clientX - panStartRef.current.x,
        y: e.clientY - panStartRef.current.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setDraggingNodeId(null);
  };

  // Filtered nodes
  const displayNodes = useMemo(() => {
    let result = nodesRef.current;
    if (filterLabel) {
      result = result.filter((n) => n.type === filterLabel);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (n) => n.label.toLowerCase().includes(q) || n.type.toLowerCase().includes(q)
      );
    }
    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodesRef.current, filterLabel, searchQuery, animationTick]);

  // Statistics calculation for Schema Buttons
  const labelCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    nodesRef.current.forEach((n) => {
      counts[n.type] = (counts[n.type] || 0) + 1;
    });
    return counts;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodesRef.current, animationTick]);

  // Relationship counts
  const relationshipCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    edgesRef.current.forEach((e) => {
      counts[e.type] = (counts[e.type] || 0) + 1;
    });
    return counts;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edgesRef.current, animationTick]);

  // Find connected neighbors for the selected node
  const selectedNeighbors = useMemo(() => {
    if (!selected) return [];
    const connectedEdges = edgesRef.current.filter(
      (e) => e.source === selected.id || e.target === selected.id
    );
    const nodeMap = new Map<string, NodeData>();
    nodesRef.current.forEach((n) => nodeMap.set(n.id, n));

    return connectedEdges.map((e) => {
      const isOut = e.source === selected.id;
      const otherId = isOut ? e.target : e.source;
      const otherNode = nodeMap.get(otherId);
      return {
        direction: isOut ? "OUTGOING" : "INCOMING",
        relType: e.type,
        targetNode: otherNode,
        targetLabel: otherNode?.label || otherId,
        targetType: otherNode?.type || "Entity",
      };
    });
  }, [selected]);

  return (
    <div className="relative min-h-screen select-none bg-[#101214] text-[#D8DADC] overflow-hidden font-sans">
      <div className="flex flex-1 flex-col h-screen relative">
        {/* ── Top Neo4j Browser Toolbar ── */}
        <div className="h-11 border-b border-[#282B30] bg-[#16171A] px-4 flex items-center justify-between z-20 shrink-0">
          <div className="flex items-center gap-3">
            {/* Neo4j Tab Switcher: [Graph] [Table] [RAW] */}
            <div className="flex rounded-md bg-[#0D0E10] p-0.5 border border-[#282B30]">
              <button
                onClick={() => setActiveTab("graph")}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded transition-all",
                  activeTab === "graph"
                    ? "bg-[#0E3D48] text-[#26C6DA] shadow-sm"
                    : "text-[#888E96] hover:text-[#D8DADC]"
                )}
              >
                Graph
              </button>
              <button
                onClick={() => setActiveTab("table")}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded transition-all",
                  activeTab === "table"
                    ? "bg-[#0E3D48] text-[#26C6DA] shadow-sm"
                    : "text-[#888E96] hover:text-[#D8DADC]"
                )}
              >
                Table
              </button>
              <button
                onClick={() => setActiveTab("raw")}
                className={cn(
                  "px-3 py-1 text-xs font-semibold rounded transition-all",
                  activeTab === "raw"
                    ? "bg-[#0E3D48] text-[#26C6DA] shadow-sm"
                    : "text-[#888E96] hover:text-[#D8DADC]"
                )}
              >
                RAW
              </button>
            </div>

            {/* Dataset Selector Dropdown */}
            {ingestedFiles.length > 0 && (
              <div className="flex items-center gap-1.5 bg-[#0D0E10] px-2.5 py-1 rounded border border-[#282B30] text-xs">
                <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <select
                  value={activeFile || ""}
                  onChange={(e) => {
                    const f = e.target.value;
                    setActiveFile(f);
                    loadGraph(f);
                  }}
                  className="bg-transparent text-white font-medium focus:outline-none cursor-pointer pr-1"
                >
                  {ingestedFiles.map((file) => (
                    <option key={file.filename} value={file.filename} className="bg-[#1A1C20] text-white">
                      {file.filename} ({file.node_count} nodes)
                    </option>
                  ))}
                </select>

                {activeFile && (
                  <button
                    onClick={() => handleDeleteFile(activeFile)}
                    title={`Delete dataset ${activeFile}`}
                    className="text-rose-400 hover:text-rose-300 ml-1 p-0.5"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                )}
              </div>
            )}

            {/* Quick status pill */}
            <div className="hidden lg:flex items-center gap-2 text-xs font-mono text-[#8C939E]">
              <span>{displayNodes.length} nodes</span>
              <span>&bull;</span>
              <span>{edgesRef.current.length} relationships</span>
            </div>
          </div>

          {/* Action Toolbar */}
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative hidden md:block">
              <Search className="h-3 w-3 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#717882]" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Find node..."
                className="pl-7 pr-3 py-1 text-xs bg-[#0D0E10] border border-[#282B30] rounded text-[#D8DADC] w-32 focus:outline-none focus:border-[#26C6DA]"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[#717882] hover:text-white text-xs"
                >
                  ✕
                </button>
              )}
            </div>

            {/* Dynamic Motion Toggle */}
            <button
              onClick={() => setMotionActive((m) => !m)}
              className={cn(
                "px-2.5 py-1 text-xs font-mono rounded border transition-colors flex items-center gap-1.5",
                motionActive
                  ? "border-[#26C6DA]/40 text-[#26C6DA] bg-[#0E3D48]/40"
                  : "border-[#282B30] text-[#717882] hover:bg-[#202226]"
              )}
              title="Toggle dynamic harmonic node motion"
            >
              <Activity className="h-3 w-3" />
              <span className="hidden sm:inline">{motionActive ? "Dynamic Motion" : "Stationary"}</span>
            </button>

            {/* CSV / Excel Ingestion */}
            <button
              onClick={() => setShowUploadModal(true)}
              className="px-2.5 py-1 text-xs font-semibold rounded border border-[#2E7D32] bg-[#1B5E20] hover:bg-[#2E7D32] text-white flex items-center gap-1.5 shadow-sm"
            >
              <FileSpreadsheet className="h-3 w-3" />
              <span className="hidden sm:inline">Upload CSV / Excel</span>
            </button>

            {/* Cypher Console Toggle (Shortened / Closed by default) */}
            <button
              onClick={() => setCypherOpen((o) => !o)}
              className={cn(
                "px-2.5 py-1 text-xs font-mono rounded border transition-colors flex items-center gap-1.5",
                cypherOpen
                  ? "border-[#00BCD4] text-[#00BCD4] bg-[#0E3D48]"
                  : "border-[#282B30] text-[#9AA0A6] hover:bg-[#202226]"
              )}
            >
              <Terminal className="h-3 w-3" />
              <span className="hidden sm:inline">Cypher Console</span>
            </button>
          </div>
        </div>

        {/* ── Schema Badges & Colored Buttons Describing Nodes, Relationships, and Attributes (Neo4j Style) ── */}
        <div className="h-10 border-b border-[#24272D] bg-[#121417] px-4 flex items-center justify-between z-15 overflow-x-auto text-xs shrink-0 gap-4">
          <div className="flex items-center gap-2 overflow-x-auto py-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#757D8A] shrink-0">
              Node Labels:
            </span>

            {/* All Filter Reset */}
            <button
              onClick={() => setFilterLabel(null)}
              className={cn(
                "px-2.5 py-0.5 rounded-full text-[11px] font-medium transition-all shrink-0",
                !filterLabel
                  ? "bg-[#282C34] text-white border border-[#3E4451]"
                  : "text-[#8C939E] hover:text-white"
              )}
            >
              All ({nodesRef.current.length})
            </button>

            {/* Colored Buttons for each Entity Node Label */}
            {Object.entries(labelCounts).map(([label, count]) => {
              const color = getNodeColor(label);
              const isActive = filterLabel === label;
              return (
                <button
                  key={label}
                  onClick={() => setFilterLabel(isActive ? null : label)}
                  className={cn(
                    "px-2.5 py-0.5 rounded-full text-[11px] font-semibold flex items-center gap-1.5 transition-all shrink-0 border",
                    isActive
                      ? "ring-2 ring-white/60 shadow-md scale-105"
                      : "opacity-85 hover:opacity-100"
                  )}
                  style={{
                    backgroundColor: color.bg,
                    borderColor: color.border,
                    color: color.text,
                  }}
                  title={`Filter by ${label} (${count} nodes)`}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-white shrink-0" />
                  <span>{label}</span>
                  <span className="text-[10px] opacity-80 font-mono">({count})</span>
                </button>
              );
            })}
          </div>

          {/* Relationship & Attribute Badges */}
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#757D8A] hidden xl:inline">
              Relationships:
            </span>
            <div className="flex items-center gap-1.5 overflow-x-auto">
              {Object.entries(relationshipCounts).slice(0, 4).map(([relType, count]) => (
                <span
                  key={relType}
                  className="px-2 py-0.5 rounded bg-[#1B1E24] border border-[#2E333D] text-[10px] font-mono text-[#A0A7B5] flex items-center gap-1 shrink-0"
                  title={`${relType} edge type`}
                >
                  <span className="text-cyan-400">&rarr;</span>
                  <span className="font-semibold text-white">{relType}</span>
                  <span className="text-[#717882]">({count})</span>
                </span>
              ))}
            </div>

            {/* Attributes Pill */}
            <button
              onClick={() => setShowPropertiesModal(true)}
              className="px-2 py-0.5 rounded bg-[#1A1F2B] border border-[#2D3748] text-[10px] font-mono text-[#63B3ED] hover:bg-[#232B3E] transition-colors flex items-center gap-1 shrink-0 ml-2"
              title="View dataset attributes & properties schema"
            >
              <Tag className="h-3 w-3" />
              <span>Attributes ({DATASET_PROPERTIES.length})</span>
            </button>
          </div>
        </div>

        {/* ── Main Canvas Viewport (Deep Solid Black Background) ── */}
        <div className="flex-1 relative w-full h-full bg-[#101214] overflow-hidden">
          {activeTab === "table" ? (
            /* Tabular View (Table Tab) */
            <div className="h-full w-full overflow-auto p-6 font-mono text-xs">
              <div className="mb-4 text-[#8C939E] text-xs flex items-center justify-between">
                <span>Showing {displayNodes.length} dataset entities from Neo4j</span>
                {activeFile && <span className="text-emerald-400 font-semibold">{activeFile}</span>}
              </div>
              <table className="w-full text-left border-collapse border border-[#282B30]">
                <thead className="bg-[#18191C] text-[#8C939E] border-b border-[#282B30]">
                  <tr>
                    <th className="p-2.5 border-r border-[#282B30]">Node ID</th>
                    <th className="p-2.5 border-r border-[#282B30]">Entity Name / Label</th>
                    <th className="p-2.5 border-r border-[#282B30]">Type / Category</th>
                    <th className="p-2.5 border-r border-[#282B30]">Degree</th>
                    <th className="p-2.5">Key Attributes</th>
                  </tr>
                </thead>
                <tbody>
                  {displayNodes.map((n) => (
                    <tr key={n.id} className="border-b border-[#202226] hover:bg-[#16171A]">
                      <td className="p-2 border-r border-[#282B30] text-[#717882]">{n.id}</td>
                      <td className="p-2 border-r border-[#282B30] font-bold text-white">{n.label}</td>
                      <td className="p-2 border-r border-[#282B30]">
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-bold"
                          style={{
                            backgroundColor: getNodeColor(n.type).bg,
                            color: getNodeColor(n.type).text,
                          }}
                        >
                          {n.type}
                        </span>
                      </td>
                      <td className="p-2 border-r border-[#282B30]">{n.degree ?? 0}</td>
                      <td className="p-2 truncate max-w-md text-[#A6ADB5]">
                        {JSON.stringify(n.properties)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : activeTab === "raw" ? (
            /* RAW JSON View (RAW Tab) */
            <div className="h-full w-full overflow-auto p-6 font-mono text-xs text-[#26C6DA]">
              <div className="mb-2 text-[#8C939E]">
                RAW Cypher Subgraph Payload ({displayNodes.length} nodes, {edgesRef.current.length} relationships)
              </div>
              <pre className="whitespace-pre-wrap bg-[#0D0E10] p-4 rounded-lg border border-[#282B30]">
                {JSON.stringify(
                  {
                    dataset: activeFile || "All Spreadsheets",
                    nodes: displayNodes.map((n) => ({
                      id: n.id,
                      label: n.label,
                      type: n.type,
                      properties: n.properties,
                    })),
                    relationships: edgesRef.current.map((e) => ({
                      source: e.source,
                      target: e.target,
                      type: e.type,
                    })),
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          ) : (
            /* Interactive Neo4j Graph Canvas (Graph Tab) */
            <div
              className="w-full h-full cursor-grab active:cursor-grabbing relative"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
            >
              {/* Empty State when no nodes are loaded */}
              {displayNodes.length === 0 && !loading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10">
                  <div className="h-16 w-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-4">
                    <FileSpreadsheet className="h-8 w-8 text-emerald-400" />
                  </div>
                  <h3 className="text-base font-bold text-white mb-2">No Spreadsheet Knowledge Graph Loaded</h3>
                  <p className="text-xs text-[#8C939E] max-w-md mb-6 leading-relaxed">
                    Upload your CSV, XLSX, or XLS dataset to generate authentic Neo4j entity nodes, dimensional clusters, and directional relationships.
                  </p>
                  <Button
                    onClick={() => setShowUploadModal(true)}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs px-5 py-2.5 rounded-lg flex items-center gap-2 shadow-lg"
                  >
                    <FileUp className="h-4 w-4" /> Upload Excel or CSV Dataset
                  </Button>
                </div>
              )}

              {/* Zoom Controls */}
              <div className="absolute bottom-6 left-6 z-20 flex flex-col gap-1 bg-[#16171A]/90 border border-[#282B30] p-1 rounded-lg shadow-xl backdrop-blur-sm">
                <button
                  onClick={handleZoomIn}
                  className="h-7 w-7 flex items-center justify-center text-[#8C939E] hover:text-white hover:bg-[#282B30] rounded"
                  title="Zoom In"
                >
                  <ZoomIn className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={handleZoomOut}
                  className="h-7 w-7 flex items-center justify-center text-[#8C939E] hover:text-white hover:bg-[#282B30] rounded"
                  title="Zoom Out"
                >
                  <ZoomOut className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={handleResetView}
                  className="h-7 w-7 flex items-center justify-center text-[#8C939E] hover:text-white hover:bg-[#282B30] rounded"
                  title="Reset View"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              </div>

              {/* Status Message Badge */}
              {statusMessage && (
                <div className="absolute bottom-6 right-6 z-20 pointer-events-none">
                  <span className="text-[11px] font-mono text-[#8C939E] bg-[#16171A]/90 border border-[#282B30] px-3 py-1.5 rounded-lg shadow-sm backdrop-blur-sm">
                    {statusMessage}
                  </span>
                </div>
              )}

              {/* SVG Surface */}
              <svg
                ref={svgRef}
                className="absolute inset-0 w-full h-full"
                viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
                preserveAspectRatio="xMidYMid slice"
              >
                <defs>
                  {/* Small, delicate Neo4j arrow markers */}
                  <marker
                    id="neo4j-small-arrow"
                    viewBox="0 0 6 6"
                    refX="5"
                    refY="3"
                    markerWidth="4"
                    markerHeight="4"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 6 3 L 0 6 z" fill="#5A6270" />
                  </marker>

                  <marker
                    id="neo4j-small-arrow-active"
                    viewBox="0 0 6 6"
                    refX="5"
                    refY="3"
                    markerWidth="4.5"
                    markerHeight="4.5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 6 3 L 0 6 z" fill="#FFFFFF" />
                  </marker>
                </defs>

                <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                  {/* ── Directed Relationships: Thin, Delicate Lines with Subtle Labels & Small Arrowheads ── */}
                  {edgesRef.current.map((edge) => {
                    const source = displayNodes.find(
                      (n) => n.id === edge.source || n.label === edge.source
                    );
                    const target = displayNodes.find(
                      (n) => n.id === edge.target || n.label === edge.target
                    );
                    if (!source || !target) return null;

                    const isHighlighted =
                      hovered === source.id ||
                      hovered === target.id ||
                      selected?.id === source.id ||
                      selected?.id === target.id;

                    // Calculate intersection at circle boundaries
                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const ux = dx / dist;
                    const uy = dy / dist;

                    const startX = source.x + ux * source.r;
                    const startY = source.y + uy * source.r;
                    const endX = target.x - ux * (target.r + 4);
                    const endY = target.y - uy * (target.r + 4);

                    const midX = (startX + endX) / 2;
                    const midY = (startY + endY) / 2;

                    return (
                      <g key={edge.id} className="transition-all duration-100">
                        {/* Thin Small Line with Delicate Arrowhead */}
                        <line
                          x1={startX}
                          y1={startY}
                          x2={endX}
                          y2={endY}
                          stroke={isHighlighted ? "#FFFFFF" : "#3B4250"}
                          strokeWidth={isHighlighted ? 1.6 : 0.9}
                          markerEnd={
                            isHighlighted
                              ? "url(#neo4j-small-arrow-active)"
                              : "url(#neo4j-small-arrow)"
                          }
                        />

                        {/* Small Relationship Label Directly on Line */}
                        <text
                          x={midX}
                          y={midY - 2.5}
                          textAnchor="middle"
                          className={cn(
                            "text-[7.5px] font-mono font-semibold tracking-tighter uppercase select-none pointer-events-none transition-colors",
                            isHighlighted ? "fill-white" : "fill-[#8C939E]"
                          )}
                        >
                          {edge.type.length > 15 ? `${edge.type.slice(0, 14)}…` : edge.type}
                        </text>
                      </g>
                    );
                  })}

                  {/* ── Neo4j Nodes: Colorful Circles with Labelling Written INSIDE ── */}
                  {displayNodes.map((node) => {
                    const isSelected = selected?.id === node.id;
                    const isHover = hovered === node.id;
                    const colorConfig = getNodeColor(node.type);
                    const { line1, line2 } = formatInnerLabel(node.label, node.r);

                    return (
                      <g
                        key={node.id}
                        transform={`translate(${node.x}, ${node.y})`}
                        onMouseEnter={() => setHovered(node.id)}
                        onMouseLeave={() => setHovered(null)}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          setDraggingNodeId(node.id);
                        }}
                        onClick={() => setSelected(node)}
                        className="cursor-pointer"
                      >
                        {/* Selection Halo */}
                        {isSelected && (
                          <circle
                            r={node.r + 5}
                            fill="none"
                            stroke="#FFFFFF"
                            strokeWidth="2"
                            strokeDasharray="4 2"
                            className="animate-spin"
                            style={{ animationDuration: "12s" }}
                          />
                        )}

                        {/* Solid Colored Circle (Neo4j Style) */}
                        <circle
                          r={node.r}
                          fill={colorConfig.bg}
                          stroke={isSelected ? "#FFFFFF" : isHover ? "#F3F4F6" : colorConfig.border}
                          strokeWidth={isSelected ? 2.5 : isHover ? 2.0 : 1.2}
                          className="drop-shadow-md transition-transform"
                        />

                        {/* Node Label Written DIRECTLY INSIDE the Circle */}
                        {line2 ? (
                          <>
                            <text
                              textAnchor="middle"
                              y="-2"
                              className="text-[9px] font-semibold select-none pointer-events-none"
                              fill={colorConfig.text}
                            >
                              {line1}
                            </text>
                            <text
                              textAnchor="middle"
                              y="9"
                              className="text-[9px] font-semibold select-none pointer-events-none"
                              fill={colorConfig.text}
                            >
                              {line2}
                            </text>
                          </>
                        ) : (
                          <text
                            textAnchor="middle"
                            y="3.5"
                            className={cn(
                              "font-bold select-none pointer-events-none",
                              node.r >= 30 ? "text-[11px]" : "text-[9.5px]"
                            )}
                            fill={colorConfig.text}
                          >
                            {line1}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>
          )}

          {/* ── Side Inspector Card (Shows Full Details of Clicked Node) ── */}
          {selected && (
            <div className="absolute top-4 right-4 z-30 w-80 bg-[#16171A]/95 backdrop-blur-md border border-[#2E3238] rounded-xl shadow-2xl p-4 space-y-3 font-sans">
              <div className="flex items-center justify-between pb-2 border-b border-[#282B30]">
                <div className="flex items-center gap-2 truncate">
                  <span
                    className="h-3 w-3 rounded-full shrink-0"
                    style={{ backgroundColor: getNodeColor(selected.type).bg }}
                  />
                  <span className="font-bold text-white text-xs truncate">
                    {selected.label}
                  </span>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-[#8C939E] hover:text-white text-xs p-1"
                >
                  ✕
                </button>
              </div>

              {/* Type & Degree */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-[#0D0E10] p-2 rounded border border-[#282B30]">
                  <span className="text-[10px] text-[#8C939E] block">Entity Type</span>
                  <span className="font-semibold text-white">{selected.type}</span>
                </div>
                <div className="bg-[#0D0E10] p-2 rounded border border-[#282B30]">
                  <span className="text-[10px] text-[#8C939E] block">Degree</span>
                  <span className="font-semibold text-white">
                    {selected.in_degree ?? 0} in &bull; {selected.out_degree ?? 0} out
                  </span>
                </div>
              </div>

              {/* Connected Relationships & Neighbors */}
              {selectedNeighbors.length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-[#8C939E] uppercase tracking-wider block mb-1">
                    Connected Relationships ({selectedNeighbors.length})
                  </span>
                  <div className="bg-[#0D0E10] p-2 rounded border border-[#282B30] max-h-28 overflow-y-auto space-y-1.5 text-[11px]">
                    {selectedNeighbors.map((nbr, idx) => (
                      <div key={idx} className="flex items-center justify-between gap-1">
                        <span className="text-cyan-400 font-mono text-[10px]">
                          {nbr.direction === "OUTGOING" ? "→" : "←"} {nbr.relType}
                        </span>
                        <span className="text-[#E0E2E5] truncate font-medium max-w-[130px]">
                          {nbr.targetLabel}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Attributes Table */}
              {selected.properties && Object.keys(selected.properties).length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-[#8C939E] uppercase tracking-wider block mb-1">
                    Attributes & Properties
                  </span>
                  <div className="bg-[#0D0E10] p-2 rounded border border-[#282B30] max-h-36 overflow-y-auto space-y-1 text-[11px] font-mono">
                    {Object.entries(selected.properties).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="text-[#8C939E] truncate">{k}:</span>
                        <span className="text-[#E0E2E5] truncate font-medium">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Cypher Quick Query */}
              <button
                onClick={() => {
                  const query = `MATCH (n {name: "${selected.label}"})-[r]-(m) RETURN n, r, m LIMIT 25`;
                  setCypherInput(query);
                  setCypherOpen(true);
                  handleRunCypher(query);
                }}
                className="w-full py-1.5 text-xs font-semibold rounded bg-[#00BCD4] hover:bg-[#26C6DA] text-[#002F35] transition-colors flex items-center justify-center gap-1.5"
              >
                <Play className="h-3 w-3" /> Query Node in Cypher
              </button>
            </div>
          )}
        </div>

        {/* ── Bottom Cypher Console Drawer (Shortened & Compact) ── */}
        {cypherOpen && (
          <div className="h-40 border-t border-[#282B30] bg-[#141518] p-3 flex flex-col gap-2 z-30 shrink-0 font-mono text-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-emerald-400 font-bold bg-[#0F2D1F] px-1.5 py-0.5 rounded border border-[#164E30]">
                  neo4j$
                </span>
                <span className="text-white font-semibold">Cypher Query Runner</span>
                <span className="text-[#717882] text-[11px] hidden sm:inline">
                  (Queries strictly against uploaded spreadsheet dataset)
                </span>
              </div>
              <button
                onClick={() => setCypherOpen(false)}
                className="text-[#8C939E] hover:text-white text-xs px-1"
                title="Close console"
              >
                ✕
              </button>
            </div>

            {/* Presets */}
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-[#717882]">Presets:</span>
              {[
                "MATCH (c:Customer)-[r:LOCATED_IN]->(city:City) RETURN c, r, city LIMIT 35",
                "MATCH (city:City)-[r:IN_COUNTRY]->(country:Country) RETURN city, r, country LIMIT 30",
                "MATCH (c:Customer)-[r:AFFILIATED_WITH]->(comp:Company) RETURN c, r, comp LIMIT 25",
                "MATCH (c:Customer)-[r:PLACED]->(o:Order) RETURN c, r, o LIMIT 30",
                "MATCH (o:Order)-[r:CONTAINS]->(i:Item) RETURN o, r, i LIMIT 30",
                "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 35",
              ].map((preset) => (
                <button
                  key={preset}
                  onClick={() => {
                    setCypherInput(preset);
                    handleRunCypher(preset);
                  }}
                  className="bg-[#1C1E22] hover:bg-[#282B30] text-[#D8DADC] px-2 py-0.5 rounded border border-[#2E3238] transition-colors text-[10px]"
                >
                  {preset.slice(0, 36)}…
                </button>
              ))}
            </div>

            {/* Input Row */}
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={cypherInput}
                onChange={(e) => setCypherInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleRunCypher();
                }}
                className="flex-1 bg-[#0D0E10] border border-[#282B30] px-3 py-1 rounded text-white focus:outline-none focus:border-[#26C6DA] text-xs"
                placeholder="MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 35"
              />
              <button
                onClick={() => handleRunCypher()}
                disabled={cypherRunning || !cypherInput.trim()}
                className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold transition-colors flex items-center gap-1 text-xs"
              >
                <Play className="h-3 w-3" /> {cypherRunning ? "Running..." : "Run"}
              </button>
            </div>

            {/* Output preview */}
            {cypherError && (
              <div className="text-rose-400 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-900 text-[11px]">
                {cypherError}
              </div>
            )}
            {cypherResult && (
              <div className="text-[#717882] text-[10px] truncate">
                Returned {cypherResult.row_count} rows in {(cypherResult.elapsed_ms || 0).toFixed(1)}ms.
              </div>
            )}
          </div>
        )}

        {/* ── Dataset Attributes & Properties Schema Modal ── */}
        {showPropertiesModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="w-full max-w-lg bg-[#16171A] border border-[#2E3238] rounded-2xl shadow-2xl overflow-hidden font-sans">
              <div className="p-4 border-b border-[#282B30] flex items-center justify-between bg-[#121316]">
                <div className="flex items-center gap-2">
                  <Tag className="h-4 w-4 text-cyan-400" />
                  <h3 className="font-bold text-white text-sm">Dataset Properties & Schema Breakdown</h3>
                </div>
                <button
                  onClick={() => setShowPropertiesModal(false)}
                  className="text-[#8C939E] hover:text-white"
                >
                  ✕
                </button>
              </div>

              <div className="p-5 space-y-4">
                <div>
                  <h4 className="text-xs font-bold text-[#8C939E] uppercase tracking-wider mb-2">
                    Node Entity Types & Visual Badges
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(NEO4J_PALETTE).map(([name, conf]) => (
                      <div
                        key={name}
                        className="p-2 rounded bg-[#0D0E10] border border-[#282B30] flex items-center gap-2"
                      >
                        <span
                          className="h-3.5 w-3.5 rounded-full shrink-0"
                          style={{ backgroundColor: conf.bg }}
                        />
                        <div>
                          <span className="text-xs font-semibold text-white block">{name}</span>
                          <span className="text-[10px] text-[#717882]">{conf.role}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-[#8C939E] uppercase tracking-wider mb-2">
                    Key Column Properties Ingested
                  </h4>
                  <div className="flex flex-wrap gap-1.5 font-mono text-xs">
                    {DATASET_PROPERTIES.map((prop) => (
                      <span
                        key={prop}
                        className="px-2 py-1 bg-[#1F2228] border border-[#2E333D] rounded text-[#E0E2E5]"
                      >
                        {prop}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-[#8C939E] uppercase tracking-wider mb-2">
                    Relationship Semantics
                  </h4>
                  <div className="space-y-1.5 text-xs font-mono">
                    {RELATIONSHIP_TYPES.map((rel) => (
                      <div
                        key={rel.name}
                        className="p-1.5 rounded bg-[#0D0E10] border border-[#282B30] flex items-center justify-between"
                      >
                        <span className="text-white font-bold">
                          ({rel.source}) -[:{rel.name}]-&gt; ({rel.target})
                        </span>
                        <span className="text-emerald-400 text-[10px]">Active</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── CSV / Excel Upload Modal ── */}
        {showUploadModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="w-full max-w-xl bg-[#16171A] border border-[#2E3238] rounded-2xl shadow-2xl overflow-hidden font-sans">
              <div className="p-4 border-b border-[#282B30] flex items-center justify-between bg-[#121316]">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-sm">Spreadsheet Knowledge Graph Manager</h3>
                </div>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-[#8C939E] hover:text-white"
                >
                  ✕
                </button>
              </div>

              <div className="p-5 space-y-4">
                {/* Dropzone */}
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files) setUploadFiles(Array.from(e.dataTransfer.files));
                  }}
                  onClick={() => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.multiple = true;
                    input.accept = ".csv,.xlsx,.xls,.tsv";
                    input.onchange = (e) => {
                      const files = (e.target as HTMLInputElement).files;
                      if (files) setUploadFiles(Array.from(files));
                    };
                    input.click();
                  }}
                  className="border-2 border-dashed border-[#2E7D32]/50 hover:border-[#2E7D32] bg-[#1B5E20]/10 rounded-xl p-6 text-center cursor-pointer transition-all"
                >
                  <FileUp className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
                  <div className="font-semibold text-xs text-white">
                    Drop CSV, XLSX, or XLS files here to generate Knowledge Graph
                  </div>
                  <p className="text-[11px] text-[#8C939E] mt-1">
                    Or click anywhere to browse from your computer
                  </p>
                </div>

                {uploadFiles.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs text-[#8C939E]">Files to Ingest ({uploadFiles.length}):</div>
                    <div className="space-y-1 max-h-24 overflow-y-auto">
                      {uploadFiles.map((f, i) => (
                        <div key={i} className="text-xs text-white bg-[#0D0E10] p-1.5 rounded flex justify-between">
                          <span>{f.name}</span>
                          <button onClick={() => setUploadFiles(uploadFiles.filter((_, idx) => idx !== i))}>✕</button>
                        </div>
                      ))}
                    </div>
                    <button
                      onClick={handleFileUpload}
                      disabled={uploadProcessing}
                      className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded transition-colors"
                    >
                      {uploadProcessing ? "Generating Graph..." : "Generate Neo4j Graph from Dataset"}
                    </button>
                  </div>
                )}

                {/* Ingested Files List (Delete Option) */}
                <div className="pt-3 border-t border-[#282B30] space-y-2">
                  <div className="text-xs font-bold text-white uppercase tracking-wider">
                    Previously Ingested Spreadsheets
                  </div>
                  {ingestedFiles.length === 0 ? (
                    <div className="text-xs text-[#8C939E]">No spreadsheets uploaded yet.</div>
                  ) : (
                    <div className="max-h-32 overflow-y-auto space-y-1">
                      {ingestedFiles.map((file) => (
                        <div
                          key={file.filename}
                          className="flex items-center justify-between p-2 rounded bg-[#0D0E10] border border-[#282B30] text-xs"
                        >
                          <div>
                            <span className="font-semibold text-white block">{file.filename}</span>
                            <span className="text-[10px] text-[#8C939E] font-mono">
                              {file.node_count} nodes &bull; {file.sample_labels.join(", ") || "Entities"}
                            </span>
                          </div>
                          <button
                            disabled={deletingFile === file.filename}
                            onClick={() => handleDeleteFile(file.filename)}
                            className="text-rose-400 hover:text-rose-300 text-xs px-2 py-1 rounded hover:bg-rose-950/40 flex items-center gap-1"
                          >
                            <Trash2 className="h-3 w-3" />
                            {deletingFile === file.filename ? "Deleting..." : "Delete"}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {uploadProgress && <div className="text-xs text-amber-400">{uploadProgress}</div>}
                {uploadSuccess && <div className="text-xs text-emerald-400">{uploadSuccess}</div>}
                {uploadError && <div className="text-xs text-rose-400">{uploadError}</div>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

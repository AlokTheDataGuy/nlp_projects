import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import { VisualizationData } from '../../services/visualization';

interface TreeGraphProps {
  data: VisualizationData | null;
  loading: boolean;
  error: string | null;
  title: string;
}

const TreeGraph: React.FC<TreeGraphProps> = ({ data, loading, error, title }) => {
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const updateDimensions = () => {
      const container = document.getElementById('tree-container');
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: container.clientHeight,
        });
      }
    };

    window.addEventListener('resize', updateDimensions);
    updateDimensions();

    return () => {
      window.removeEventListener('resize', updateDimensions);
    };
  }, []);

  useEffect(() => {
    // When data changes, apply tree layout
    if (graphRef.current && data) {
      // Disable physics simulation for tree layout
      graphRef.current.d3Force('charge', null);
      graphRef.current.d3Force('center', null);
      graphRef.current.d3Force('link').distance(() => 80);

      // Apply radial force to create tree-like structure
      const rootId = data.nodes[0]?.id.toString();

      // Custom force to position nodes in a tree layout
      const forceTree = () => {
        const nodes = graphRef.current.graphData().nodes;
        const links = graphRef.current.graphData().links;

        // Create a map of node id to level (distance from root)
        const levels = new Map();
        levels.set(rootId, 0);

        // BFS to assign levels
        const queue = [rootId];
        while (queue.length > 0) {
          const nodeId = queue.shift();
          const level = levels.get(nodeId);

          // Find children
          links
            .filter((link: any) => link.source.id === nodeId || (typeof link.source === 'string' && link.source === nodeId))
            .forEach((link: any) => {
              const childId = typeof link.target === 'string' ? link.target : link.target.id;
              if (!levels.has(childId)) {
                levels.set(childId, level + 1);
                queue.push(childId);
              }
            });
        }

        // Position nodes based on levels
        nodes.forEach((node: any) => {
          const level = levels.get(node.id) || 0;
          const angle = (node.index || 0) * 2 * Math.PI / nodes.length;

          // Radial layout
          const radius = 100 * (level + 1);
          node.x = radius * Math.cos(angle);
          node.y = radius * Math.sin(angle);

          // Apply some force to keep nodes at their level
          const dx = node.x - node.vx;
          const dy = node.y - node.vy;
          node.vx += dx * 0.1;
          node.vy += dy * 0.1;
        });
      };

      graphRef.current.d3Force('tree', forceTree);
      graphRef.current.d3ReheatSimulation();
    }
  }, [data]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (!data || !data.nodes.length) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography>No data available for visualization.</Typography>
      </Box>
    );
  }

  // Convert data to format expected by ForceGraph
  const graphData = {
    nodes: data.nodes.map((node, index) => ({
      ...node,
      id: node.id.toString(),
      val: node.size,
      index
    })),
    links: data.links.map(link => ({
      ...link,
      source: link.source.toString(),
      target: link.target.toString()
    }))
  };

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <Box id="tree-container" sx={{ flex: 1, minHeight: 500 }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel="name"
          nodeColor={(node: any) => `hsl(${node.group * 60}, 70%, 50%)`}
          nodeVal={(node: any) => node.val}
          linkLabel={(link: any) => link.relation}
          linkWidth={(link: any) => Math.sqrt(link.value)}
          linkDirectionalArrowLength={6}
          linkDirectionalArrowRelPos={0.8}
          cooldownTicks={100}
          onEngineStop={() => graphRef.current?.zoomToFit(400)}
        />
      </Box>
    </Paper>
  );
};

export default TreeGraph;

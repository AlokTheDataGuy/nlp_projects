import React, { useRef, useEffect, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import { VisualizationData } from '../../services/visualization';
import SpriteText from 'three-spritetext';

interface ForceGraph3DProps {
  data: VisualizationData | null;
  loading: boolean;
  error: string | null;
  title: string;
}

const ForceGraph3DComponent: React.FC<ForceGraph3DProps> = ({ data, loading, error, title }) => {
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const updateDimensions = () => {
      const container = document.getElementById('graph-container-3d');
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
    // When data changes, reheat the simulation
    if (graphRef.current && data) {
      graphRef.current.d3Force('charge').strength(-120);
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

  // Convert data to format expected by ForceGraph3D
  const graphData = {
    nodes: data.nodes.map(node => ({
      ...node,
      id: node.id.toString(),
      val: node.size
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
      <Box id="graph-container-3d" sx={{ flex: 1, minHeight: 500 }}>
        <ForceGraph3D
          ref={graphRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel="name"
          nodeColor={(node: any) => `hsl(${node.group * 60}, 70%, 50%)`}
          nodeVal={(node: any) => node.val / 2}
          linkLabel={(link: any) => link.relation}
          linkWidth={(link: any) => Math.sqrt(link.value) * 0.5}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={0.8}
          cooldownTicks={100}
          nodeThreeObject={(node: any) => {
            const sprite = new SpriteText(node.name);
            sprite.color = `hsl(${node.group * 60}, 70%, 50%)`;
            sprite.textHeight = 8;
            return sprite;
          }}
          onEngineStop={() => graphRef.current?.zoomToFit(400, 100)}
        />
      </Box>
    </Paper>
  );
};

export default ForceGraph3DComponent;

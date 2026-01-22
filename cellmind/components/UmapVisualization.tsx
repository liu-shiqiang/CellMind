
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { CellCluster } from '../types';

interface Props {
  clusters: CellCluster[];
  onSelectCluster: (cluster: CellCluster) => void;
}

export const UmapVisualization: React.FC<Props> = ({ clusters, onSelectCluster }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || clusters.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    const allPoints = clusters.flatMap(c => c.embedding.map(p => ({ x: p[0], y: p[1], clusterId: c.id })));
    
    const xScale = d3.scaleLinear()
      .domain([d3.min(allPoints, d => d.x) || -5, d3.max(allPoints, d => d.x) || 5])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([d3.min(allPoints, d => d.y) || -5, d3.max(allPoints, d => d.y) || 5])
      .range([height - margin.bottom, margin.top]);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .style("width", "100%")
       .style("height", "auto");

    svg.append("g")
      .selectAll("circle")
      .data(allPoints)
      .join("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 3)
      .attr("fill", d => color(d.clusterId))
      .attr("opacity", 0.6)
      .on("click", (event, d) => {
        const cluster = clusters.find(c => c.id === d.clusterId);
        if (cluster) onSelectCluster(cluster);
      })
      .style("cursor", "pointer");

    // Add labels
    clusters.forEach(c => {
      const avgX = d3.mean(c.embedding, p => p[0]) || 0;
      const avgY = d3.mean(c.embedding, p => p[1]) || 0;
      svg.append("text")
        .attr("x", xScale(avgX))
        .attr("y", yScale(avgY))
        .attr("text-anchor", "middle")
        .attr("font-size", "10px")
        .attr("font-weight", "bold")
        .attr("fill", "#334155")
        .text(c.suggestedType || `C${c.id}`);
    });

  }, [clusters, onSelectCluster]);

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm relative">
      <h3 className="text-sm font-semibold mb-2 text-slate-500 uppercase tracking-wider">scGPT Embedding Visualization (UMAP)</h3>
      <svg ref={svgRef} className="w-full h-full min-h-[300px]" />
      <div className="absolute bottom-4 right-4 text-[10px] text-slate-400">
        Click a cell cluster to view details
      </div>
    </div>
  );
};

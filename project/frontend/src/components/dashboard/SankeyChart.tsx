import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';

interface SankeyNode {
  id: string;
  name: string;
  category: string;
  subcategory?: string; 
  importance?: 'high' | 'medium' | 'low'; // Add importance for visual emphasis
  icon?: string; // Optional icon identifier
}

interface SankeyLink {
  source: string;
  target: string;
  value: number;
  type?: string;
  quality?: number; // Quality score between 0-100 for additional visual encoding
}

interface SankeyChartProps {
  data: {
    nodes: SankeyNode[];
    links: SankeyLink[];
  };
  width?: number;
  height?: number;
  title?: string;
  showLegend?: boolean;
  colorScheme?: 'vibrant' | 'pastel' | 'monochrome' | 'divergent';
}

const SankeyChart: React.FC<SankeyChartProps> = ({ 
  data, 
  width = 900, 
  height = 600, 
  title = "Customer Journey Flow",
  showLegend = true,
  colorScheme = 'vibrant'
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length || !data.links.length) return;

    // Validate data - check for missing nodes
    const nodeIds = new Set(data.nodes.map(d => d.id));
    const missingNodeIds: string[] = [];
    
    data.links.forEach(link => {
      if (typeof link.source === 'string' && !nodeIds.has(link.source)) {
        missingNodeIds.push(link.source);
      }
      if (typeof link.target === 'string' && !nodeIds.has(link.target)) {
        missingNodeIds.push(link.target);
      }
    });

    if (missingNodeIds.length > 0) {
      console.error('Missing nodes in Sankey chart:', [...new Set(missingNodeIds)]);
      return; // Prevent rendering with invalid data
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom - (showLegend ? 60 : 0);

    // Add background pattern
    const defs = svg.append("defs");
    
    // Add subtle background pattern
    const pattern = defs.append("pattern")
      .attr("id", "bg-pattern")
      .attr("width", 10)
      .attr("height", 10)
      .attr("patternUnits", "userSpaceOnUse")
      .attr("patternTransform", "rotate(45)");
    
    pattern.append("rect")
      .attr("width", 10)
      .attr("height", 10)
      .attr("fill", "#f9fafb");
    
    pattern.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", 10)
      .attr("stroke", "#f3f4f6")
      .attr("stroke-width", 1);

    // Apply background pattern
    svg.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "url(#bg-pattern)");
    
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create sankey generator with adjusted settings for complexity
    const sankeyGenerator = sankey<SankeyNode, SankeyLink>()
      .nodeId(d => d.id)
      .nodeWidth(22) // Increased width
      .nodePadding(10) // Increased padding for more space between nodes
      .extent([[1, 1], [innerWidth - 1, innerHeight - 6]]);

    // Process data with validated nodes and links
    const graph = sankeyGenerator({
      nodes: data.nodes.map(d => ({ ...d })),
      links: data.links.map(d => ({ ...d }))
    });

    // Color schemes based on preference
    const colorSchemes = {
      vibrant: {
        categories: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA931', '#7E78D2'],
        links: {
          high_value: '#FF5E5B',
          medium_value: '#FFBF69',
          standard: '#4D9DE0',
          conversion: '#3FA796', 
          retention: '#7E78D2'
        }
      },
      pastel: {
        categories: ['#FFADAD', '#FFD6A5', '#FDFFB6', '#CAFFBF', '#9BF6FF'],
        links: {
          high_value: '#FEC5BB',
          medium_value: '#FAE1DD',
          standard: '#E8E8E4',
          conversion: '#D8E2DC', 
          retention: '#ECE4DB'
        }
      },
      monochrome: {
        categories: ['#CDDAFD', '#B8C0FF', '#A3ACF0', '#8D95EB', '#7882E0'],
        links: {
          high_value: '#595260',
          medium_value: '#797596',
          standard: '#958DB8',
          conversion: '#B8B3D0', 
          retention: '#DCD5E9'
        }
      },
      divergent: {
        categories: ['#3A86FF', '#61A5FF', '#80CAFF', '#FFBE0B', '#FB5607'],
        links: {
          high_value: '#FF006E',
          medium_value: '#8338EC',
          standard: '#3A86FF',
          conversion: '#38B000', 
          retention: '#FFBE0B'
        }
      }
    };

    // Select the color scheme based on the prop
    const selectedScheme = colorSchemes[colorScheme] || colorSchemes.vibrant;

    // Rich color scales for different categories and link types
    const categoryColorScale = d3.scaleOrdinal<string>()
      .domain(['awareness', 'interest', 'consideration', 'purchase', 'retention'])
      .range(selectedScheme.categories);
    
    // Different color palette for links based on value ranges
    const linkValueScale = d3.scaleSequential()
      .domain([
        d3.min(graph.links, d => d.value) || 0, 
        d3.max(graph.links, d => d.value) || 1000
      ])
      .interpolator(d3.interpolateRgb('#E0FBFC', '#EE6C4D'));

    // Link type colors
    const linkTypeColors = selectedScheme.links;

    // Draw gradient definitions for links
    graph.links.forEach((link, i) => {
      const sourceNode = link.source as any;
      const targetNode = link.target as any;
      const gradientId = `link-gradient-${i}`;
      
      const gradient = defs.append("linearGradient")
        .attr("id", gradientId)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", sourceNode.x1)
        .attr("x2", targetNode.x0);
        
      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", categoryColorScale(sourceNode.category));
        
      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", categoryColorScale(targetNode.category));
        
      // Add intermediate stop for more complex gradient
      if (link.quality !== undefined) {
        const qualityColor = d3.interpolateRgb(
          categoryColorScale(sourceNode.category),
          categoryColorScale(targetNode.category)
        )(0.5);
        
        gradient.append("stop")
          .attr("offset", "50%")
          .attr("stop-color", d3.color(qualityColor)?.brighter(link.quality / 50) || qualityColor);
      }
    });

    // Add drop shadow filter
    const filter = defs.append("filter")
      .attr("id", "drop-shadow")
      .attr("height", "130%");
    
    filter.append("feGaussianBlur")
      .attr("in", "SourceAlpha")
      .attr("stdDeviation", 3)
      .attr("result", "blur");
    
    filter.append("feOffset")
      .attr("in", "blur")
      .attr("dx", 2)
      .attr("dy", 2)
      .attr("result", "offsetBlur");
    
    const feComponentTransfer = filter.append("feComponentTransfer")
      .attr("in", "offsetBlur")
      .attr("result", "linearSlope");
      
    feComponentTransfer.append("feFuncA")
      .attr("type", "linear")
      .attr("slope", 0.3);
    
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode")
      .attr("in", "offsetBlur");
    feMerge.append("feMergeNode")
      .attr("in", "SourceGraphic");

    // Draw links with advanced styling
    g.append("g")
  .selectAll("path")
  .data(graph.links)
  .join("path")
  .attr("d", sankeyLinkHorizontal())
  .attr("stroke", (d, i) => {
    // Choose color based on link type or value
    if (d.type && linkTypeColors[d.type as keyof typeof linkTypeColors]) {
      return linkTypeColors[d.type as keyof typeof linkTypeColors];
    }
    
    return `url(#link-gradient-${i})`;
  })
  .attr("stroke-opacity", 0.7)
  .attr("stroke-width", d => Math.max(1, d.width || 0)) // Always use actual width
  .attr("fill", "none")
  .style("filter", "url(#drop-shadow)")
  .on("mouseover", function(event, d) {
    // Highlight the path with slightly increased opacity but same width
    d3.select(this)
      .attr("stroke-opacity", 0.9)
      .style("filter", "drop-shadow(0px 3px 5px rgba(0,0,0,0.3))");
    // Don't increase width on hover, just keep actual girth
        
        // Enhanced tooltip with more data
        const tooltip = d3.select("body")
          .append("div")
          .attr("class", "sankey-tooltip")
          .style("position", "absolute")
          .style("background", "rgba(15, 23, 42, 0.95)")
          .style("color", "white")
          .style("padding", "12px")
          .style("border-radius", "8px")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000")
          .style("box-shadow", "0 4px 8px rgba(0, 0, 0, 0.2)")
          .style("border-left", `4px solid ${d.type ? linkTypeColors[d.type as keyof typeof linkTypeColors] : '#4361ee'}`);

        const sourceNode = d.source as any;
        const targetNode = d.target as any;
        const conversionRate = ((d.value / (sourceNode.value || 1)) * 100).toFixed(1);
        
        tooltip.html(`
          <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">
            ${sourceNode.name} → ${targetNode.name}
          </div>
          <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: ${d.type ? linkTypeColors[d.type as keyof typeof linkTypeColors] : '#4361ee'}; margin-right: 6px;"></div>
            <span style="font-size: 12px;">Flow: ${d.value.toLocaleString()} customers</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 4px;">
            <div style="font-size: 12px;">Conversion: ${conversionRate}%</div>
            <div style="height: 4px; width: 100%; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
              <div style="height: 100%; width: ${Math.min(Number(conversionRate), 100)}%; background: ${d.type ? linkTypeColors[d.type as keyof typeof linkTypeColors] : '#4361ee'};"></div>
            </div>
          </div>
          ${d.type ? `<div style="margin-top: 6px; font-size: 11px; opacity: 0.8;">Type: ${d.type.replace('_', ' ')}</div>` : ''}
          ${d.quality ? `<div style="margin-top: 6px; font-size: 11px; opacity: 0.8;">Quality: ${d.quality}/100</div>` : ''}
        `);

        tooltip
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");
      })
      .on("mouseout", function() {
    d3.select(this)
      .attr("stroke-opacity", 0.7)
      .style("filter", "url(#drop-shadow)");
    d3.selectAll(".sankey-tooltip").remove();
  });

    // Draw nodes with enhanced styling
    const node = g.append("g")
      .selectAll("g")
      .data(graph.nodes)
      .join("g")
      .attr("transform", d => `translate(${d.x0},${d.y0})`);
    
    // Node rectangle background
    node.append("rect")
  .attr("height", d => (d.y1 || 0) - (d.y0 || 0))
  .attr("width", d => (d.x1 || 0) - (d.x0 || 0))
  .attr("fill", d => {
    // Apply different brightness based on importance
    const baseColor = categoryColorScale(d.category);
    if (d.importance === 'high') {
      return d3.color(baseColor)?.brighter(0.2) || baseColor;
    } else if (d.importance === 'low') {
      return d3.color(baseColor)?.darker(0.2) || baseColor;
    }
    return baseColor;
  })
  .attr("fill-opacity", 0.9) // Always use enhanced opacity
  .attr("stroke", d => d3.color(categoryColorScale(d.category))?.darker(0.5) || "#1a1a1a")
  .attr("stroke-width", d => d.importance === 'high' ? 2 : 1) // Always use enhanced stroke width
  .attr("rx", 6) 
  .attr("ry", 6)
  .style("filter", "drop-shadow(0px 4px 6px rgba(0,0,0,0.3))") // Always use enhanced shadow
  .on("mouseover", function(event, d) {
    // Don't change appearance on hover, just show tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "sankey-tooltip")
          .style("position", "absolute")
          .style("background", "rgba(15, 23, 42, 0.95)")
          .style("color", "white")
          .style("padding", "12px")
          .style("border-radius", "8px")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000")
          .style("box-shadow", "0 4px 8px rgba(0, 0, 0, 0.2)")
          .style("border-left", `4px solid ${categoryColorScale(d.category)}`);

        // Calculate inflow and outflow
        const totalInflow = graph.links.filter(l => (l.target as any).id === d.id)
          .reduce((sum, link) => sum + link.value, 0);
        const totalOutflow = graph.links.filter(l => (l.source as any).id === d.id)
          .reduce((sum, link) => sum + link.value, 0);
        
        // Determine if this is source, target or middle node
        let nodeType = "Transit";
        if (totalInflow === 0) nodeType = "Source";
        if (totalOutflow === 0) nodeType = "Target";
        
        // Calculate retention rate if applicable
        const retentionRate = nodeType === "Transit" ? 
          ((totalOutflow / totalInflow) * 100).toFixed(1) + "%" : "N/A";

        tooltip.html(`
          <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">${d.name}</div>
          <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: ${categoryColorScale(d.category)}; margin-right: 6px;"></div>
            <span style="font-size: 12px;">Total: ${(d.value || 0).toLocaleString()} customers</span>
          </div>
          <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12px;">
            <div>Category:</div>
            <div>${d.category.charAt(0).toUpperCase() + d.category.slice(1)}</div>
            ${d.subcategory ? `<div>Type:</div><div>${d.subcategory}</div>` : ''}
            <div>Node Type:</div>
            <div>${nodeType}</div>
            ${d.importance ? `<div>Importance:</div><div>${d.importance.charAt(0).toUpperCase() + d.importance.slice(1)}</div>` : ''}
            ${totalInflow > 0 ? `<div>Inflow:</div><div>${totalInflow.toLocaleString()}</div>` : ''}
            ${totalOutflow > 0 ? `<div>Outflow:</div><div>${totalOutflow.toLocaleString()}</div>` : ''}
            ${nodeType === "Transit" ? `<div>Retention:</div><div>${retentionRate}</div>` : ''}
          </div>
        `);

        tooltip
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");
      })
      .on("mouseout", function() {
    // Just remove tooltip, don't change appearance
    d3.selectAll(".sankey-tooltip").remove();
  });

    // Add node decorations for importance
    node.filter(d => d.importance === 'high')
      .append("rect")
      .attr("x", 2)
      .attr("y", 2)
      .attr("width", d => (d.x1 || 0) - (d.x0 || 0) - 4)
      .attr("height", 3)
      .attr("fill", "rgba(255,255,255,0.4)")
      .attr("rx", 1)
      .attr("ry", 1);

    // Node labels with improved styling
    node.append("text")
      .attr("x", d => d.x0 < innerWidth / 2 ? (d.x1 || 0) - (d.x0 || 0) + 6 : -6)
      .attr("y", d => ((d.y1 || 0) - (d.y0 || 0)) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", d => d.x0 < innerWidth / 2 ? "start" : "end")
      .attr("font-family", "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif")
      .attr("font-size", d => d.importance === 'high' ? "12px" : "11px")
      .attr("font-weight", d => d.importance === 'high' ? "600" : "500")
      .attr("fill", d => {
        const luminance = d3.hsl(categoryColorScale(d.category)).l;
        return luminance > 0.5 ? "#333333" : "#ffffff";
      })
      .text(d => d.name)
      .style("text-shadow", "0px 1px 2px rgba(0,0,0,0.1)")
      .style("pointer-events", "none");

    // Add value labels to nodes with significant values
    node.filter(d => (d.value || 0) > 1000 || d.importance === 'high')
      .append("text")
      .attr("x", d => d.x0 < innerWidth / 2 ? (d.x1 || 0) - (d.x0 || 0) + 6 : -6)
      .attr("y", d => ((d.y1 || 0) - (d.y0 || 0)) / 2 + 14)
      .attr("dy", "0.35em")
      .attr("text-anchor", d => d.x0 < innerWidth / 2 ? "start" : "end")
      .attr("font-family", "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif")
      .attr("font-size", "9px")
      .attr("fill", d => {
        const luminance = d3.hsl(categoryColorScale(d.category)).l;
        return luminance > 0.5 ? "#555555" : "#eeeeee";
      })
      .text(d => `${(d.value || 0).toLocaleString()}`)
      .style("pointer-events", "none");

    // Add icons to nodes if specified
    node.filter(d => d.icon)
      .append("text")
      .attr("x", d => d.x0 < innerWidth / 2 ? 5 : (d.x1 || 0) - (d.x0 || 0) - 5)
      .attr("y", 10)
      .attr("text-anchor", d => d.x0 < innerWidth / 2 ? "start" : "end")
      .attr("font-family", "FontAwesome")
      .attr("font-size", "10px")
      .attr("fill", d => {
        const luminance = d3.hsl(categoryColorScale(d.category)).l;
        return luminance > 0.5 ? "#333333" : "#ffffff";
      })
      .text(d => {
        // Map icon names to unicode - this would need FontAwesome or similar font
        const iconMap: {[key: string]: string} = {
          "user": "\uf007",
          "dollar": "\uf155",
          "cart": "\uf07a",
          "check": "\uf00c",
          "mail": "\uf0e0",
          "phone": "\uf095",
          "heart": "\uf004"
        };
        return iconMap[d.icon || ""] || "";
      })
      .style("pointer-events", "none");

    // Add title with enhanced styling
    if (title) {
      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-family", "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif")
        .attr("font-size", "18px")
        .attr("font-weight", "600")
        .attr("fill", "#1F2937")
        .text(title);
    }

    // Add enhanced legend
    if (showLegend) {
      // Category legend
      const legendData = ['awareness', 'interest', 'consideration', 'purchase', 'retention'];
      const legendLabels = ['Awareness', 'Interest', 'Consideration', 'Purchase', 'Retention'];
      
      const categoryLegend = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${height - 50})`);
      
      const categoryLegendItems = categoryLegend.selectAll("g")
        .data(legendData)
        .join("g")
        .attr("transform", (d, i) => `translate(${i * (innerWidth / legendData.length)}, 0)`);
      
      categoryLegendItems.append("rect")
        .attr("width", 12)
        .attr("height", 12)
        .attr("fill", d => categoryColorScale(d))
        .attr("rx", 2)
        .attr("ry", 2)
        .style("filter", "drop-shadow(0px 1px 1px rgba(0,0,0,0.1))");
      
      categoryLegendItems.append("text")
        .attr("x", 16)
        .attr("y", 10)
        .attr("font-family", "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif")
        .attr("font-size", "10px")
        .attr("fill", "#4B5563")
        .text((d, i) => legendLabels[i]);

      // Link type legend
      const linkTypes = Object.keys(linkTypeColors);
      const linkTypeLabels = linkTypes.map(t => t.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()));
      
      const linkLegend = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${height - 30})`);
      
      const linkLegendItems = linkLegend.selectAll("g")
        .data(linkTypes)
        .join("g")
        .attr("transform", (d, i) => `translate(${i * (innerWidth / linkTypes.length)}, 0)`);
      
      linkLegendItems.append("line")
        .attr("x1", 0)
        .attr("y1", 6)
        .attr("x2", 12)
        .attr("y2", 6)
        .attr("stroke", d => linkTypeColors[d as keyof typeof linkTypeColors])
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.8);
      
      linkLegendItems.append("text")
        .attr("x", 16)
        .attr("y", 10)
        .attr("font-family", "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif")
        .attr("font-size", "10px")
        .attr("fill", "#4B5563")
        .text((d, i) => linkTypeLabels[i]);
    }

  }, [data, width, height, title, showLegend, colorScheme]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ 
        background: 'white', 
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08)'
      }}
    />
  );
};

export default SankeyChart;
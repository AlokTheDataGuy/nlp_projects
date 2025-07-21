import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import yaml
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ConceptVisualizer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.viz_config = self.config['visualization']
        
    def create_concept_network(self, concept_graph, title="CS Concept Network"):
        """
        Create interactive concept network visualization
        """
        if not concept_graph.nodes():
            return go.Figure().add_annotation(text="No concept data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Calculate layout
        layout = nx.spring_layout(concept_graph, k=1, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in concept_graph.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Size based on node weight (frequency)
            weight = concept_graph.nodes[node].get('weight', 1)
            node_size.append(max(10, min(50, weight * 3)))
            node_color.append(weight)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in concept_graph.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = concept_graph.edges[edge].get('weight', 1)
            edge_weights.append(weight)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale=self.viz_config['color_scheme'],
                showscale=True,
                colorbar=dict(title="Frequency"),
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>Frequency: %{marker.color}<extra></extra>',
            name='Concepts'
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size = frequency, Connections = co-occurrence",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="grey", size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_topic_evolution(self, papers_df, title="Research Topics Over Time"):
        """
        Create visualization showing evolution of research topics over time
        """
        if papers_df.empty:
            return go.Figure().add_annotation(text="No data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Group by year and primary category
        yearly_topics = papers_df.groupby(['year', 'primary_category']).size().unstack(fill_value=0)
        
        # Create stacked area chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, topic in enumerate(yearly_topics.columns):
            fig.add_trace(go.Scatter(
                x=yearly_topics.index,
                y=yearly_topics[topic],
                mode='lines',
                stackgroup='one',
                name=topic,
                line=dict(color=colors[i % len(colors)]),
                hovertemplate=f'<b>{topic}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Number of Papers",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_author_collaboration(self, papers_df, min_papers=2):
        """
        Create author collaboration network
        """
        if papers_df.empty:
            return go.Figure().add_annotation(text="No data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Build collaboration graph
        G = nx.Graph()
        author_paper_count = defaultdict(int)
        
        for _, paper in papers_df.iterrows():
            authors = paper['authors']
            if len(authors) < 2:
                continue
            
            # Count papers per author
            for author in authors:
                author_paper_count[author] += 1
            
            # Add collaboration edges
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
        
        # Filter authors with minimum paper count
        prolific_authors = {author for author, count in author_paper_count.items() 
                          if count >= min_papers}
        
        # Create subgraph with prolific authors only
        G_filtered = G.subgraph(prolific_authors).copy()
        
        if not G_filtered.nodes():
            return go.Figure().add_annotation(text="No collaboration data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Calculate layout
        pos = nx.spring_layout(G_filtered, k=2, iterations=50)
        
        # Prepare visualization data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G_filtered.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.split()[-1])  # Use last name
            node_size.append(author_paper_count[node] * 5)
        
        edge_x = []
        edge_y = []
        
        for edge in G_filtered.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
            hoverinfo='none',
            mode='lines',
            name='Collaborations'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Papers: %{marker.size}<extra></extra>',
            name='Authors'
        ))
        
        fig.update_layout(
            title="Author Collaboration Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_keyword_trends(self, papers_df, keywords_list, title="Keyword Trends Over Time"):
        """
        Create trends for specific keywords over time
        """
        if papers_df.empty or not keywords_list:
            return go.Figure().add_annotation(text="No trend data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Create keyword trends
        keyword_trends = defaultdict(lambda: defaultdict(int))
        
        for _, paper in papers_df.iterrows():
            year = paper['year']
            text = f"{paper['title']} {paper['abstract']}".lower()
            
            for keyword in keywords_list:
                if keyword.lower() in text:
                    keyword_trends[keyword][year] += 1
        
        # Convert to DataFrame
        trend_data = []
        for keyword, yearly_counts in keyword_trends.items():
            for year, count in yearly_counts.items():
                trend_data.append({'keyword': keyword, 'year': year, 'count': count})
        
        if not trend_data:
            return go.Figure().add_annotation(text="No keyword trends found", 
                                           showarrow=False, x=0.5, y=0.5)
        
        trends_df = pd.DataFrame(trend_data)
        
        # Create line plot
        fig = px.line(trends_df, x='year', y='count', color='keyword',
                     title=title, markers=True)
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Papers Mentioning Keyword",
            height=500
        )
        
        return fig
    
    def create_category_distribution(self, papers_df, title="Research Category Distribution"):
        """
        Create category distribution visualization
        """
        if papers_df.empty:
            return go.Figure().add_annotation(text="No category data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        category_counts = papers_df['primary_category'].value_counts()
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hovertemplate='<b>%{label}</b><br>Papers: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title=title, height=500)
        
        return fig
    
    def create_paper_metrics_dashboard(self, papers_df):
        """
        Create comprehensive paper metrics dashboard
        """
        if papers_df.empty:
            return go.Figure().add_annotation(text="No metrics data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Papers per Year', 'Abstract Length Distribution', 
                          'Author Count Distribution', 'Monthly Publication Trends'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Papers per year
        yearly_counts = papers_df['year'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=yearly_counts.index, y=yearly_counts.values, 
                      mode='lines+markers', name='Papers/Year'),
            row=1, col=1
        )
        
        # Abstract length distribution
        fig.add_trace(
            go.Histogram(x=papers_df['abstract_length'], nbinsx=30, 
                        name='Abstract Length', showlegend=False),
            row=1, col=2
        )
        
        # Author count distribution
        author_counts = papers_df['author_count'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=author_counts.index, y=author_counts.values, 
                  name='Author Count', showlegend=False),
            row=2, col=1
        )
        
        # Monthly trends
        monthly_counts = papers_df['month'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=monthly_counts.index, y=monthly_counts.values, 
                      mode='lines+markers', name='Papers/Month', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Paper Metrics Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_embedding_visualization(self, embeddings, papers_df, method='tsne'):
        """
        Create 2D visualization of paper embeddings
        """
        if embeddings is None or len(embeddings) == 0:
            return go.Figure().add_annotation(text="No embedding data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Reduce dimensionality
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        # Sample data if too large for t-SNE
        if len(embeddings) > 1000 and method.lower() == 'tsne':
            sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_sample = embeddings[sample_indices]
            papers_sample = papers_df.iloc[sample_indices]
        else:
            embeddings_sample = embeddings
            papers_sample = papers_df
        
        # Reduce dimensions
        coords_2d = reducer.fit_transform(embeddings_sample)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by category
        categories = papers_sample['primary_category'].unique()
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(categories):
            mask = papers_sample['primary_category'] == category
            
            fig.add_trace(go.Scatter(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode='markers',
                name=category,
                marker=dict(
                    color=colors[i % len(colors)],
                    size=5,
                    opacity=0.7
                ),
                text=papers_sample[mask]['title'].str[:50] + '...',
                hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Paper Embeddings Visualization ({method.upper()})",
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def create_wordcloud_figure(self, text_data, title="Word Cloud"):
        """
        Create word cloud visualization
        """
        if not text_data:
            return go.Figure().add_annotation(text="No text data available", 
                                           showarrow=False, x=0.5, y=0.5)
        
        # Combine all text
        combined_text = ' '.join(text_data)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        # Convert to plotly figure
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=wordcloud.to_image(),
                xref="x", yref="y",
                x=0, y=0,
                sizex=1, sizey=1,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig

/**
 * GraphWebView — offline vis-network mind-map of the reasoning graph.
 *
 * The vis-network standalone UMD build is vendored at
 * assets/vendor/vis-network.min.txt (metro treats .txt as an asset). At
 * runtime we download the bundled asset to disk, read it as a string, and
 * inline it into the WebView HTML — no CDN, no network access required.
 *
 * Drug node taps bridge back to RN via window.ReactNativeWebView.postMessage.
 */
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';

import type { GraphEdge, GraphNode } from '@/api/osler';
import { colors, fonts, spacing } from '@/theme/tokens';

interface Props {
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  /** called when the user taps a drug node ("drug:<name>") */
  onDrugTap?: (drug: string) => void;
}

let visJsCache: string | null = null;

async function loadVisJs(): Promise<string> {
  if (visJsCache) return visJsCache;
  const asset = Asset.fromModule(
    // Vendored vis-network standalone UMD (treated as an asset via metro assetExts)
    require('@/assets/vendor/vis-network.min.txt'),
  );
  await asset.downloadAsync();
  const js = await FileSystem.readAsStringAsync(asset.localUri!);
  visJsCache = js;
  return js;
}

/** Escape the one sequence that could prematurely close our inline script. */
function safeJson(value: unknown): string {
  return JSON.stringify(value).replace(/<\/script>/gi, '<\\/script>');
}

function buildHtml(
  visJs: string,
  graph: { nodes: GraphNode[]; edges: GraphEdge[] },
): string {
  const groups = {
    patient: { color: { background: '#0F2B5B', border: '#0F2B5B' }, font: { color: '#FFFFFF' } },
    indication: { color: { background: '#1A3F7A', border: '#1A3F7A' }, font: { color: '#FFFFFF' } },
    disease: { color: { background: '#E8724A', border: '#E8724A' }, font: { color: '#FFFFFF' } },
    pathology: { color: { background: '#D97706', border: '#D97706' }, font: { color: '#FFFFFF' } },
    symptom: { color: { background: '#8C9BB5', border: '#8C9BB5' }, font: { color: '#FFFFFF' } },
    target: { color: { background: '#0D9488', border: '#0D9488' }, font: { color: '#FFFFFF' } },
    flag: { color: { background: '#DC2626', border: '#DC2626' }, font: { color: '#FFFFFF' } },
    'drug-ok': { color: { background: '#059669', border: '#059669' }, font: { color: '#FFFFFF' } },
    'drug-show': { color: { background: '#059669', border: '#059669' }, font: { color: '#FFFFFF' } },
    'drug-caution': { color: { background: '#D97706', border: '#D97706' }, font: { color: '#FFFFFF' } },
    'drug-warn': { color: { background: '#D97706', border: '#D97706' }, font: { color: '#FFFFFF' } },
    'drug-adjust': { color: { background: '#0D9488', border: '#0D9488' }, font: { color: '#FFFFFF' } },
    'drug-avoid': { color: { background: '#DC2626', border: '#DC2626' }, font: { color: '#FFFFFF' } },
    'drug-block': { color: { background: '#DC2626', border: '#DC2626' }, font: { color: '#FFFFFF' } },
  };

  return `<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<script>${visJs}</script>
<style>html,body{margin:0;padding:0;overflow:hidden;background:#F8F9FC}</style>
</head>
<body>
<div id="net" style="width:100vw;height:100vh"></div>
<script>
(function(){
  var graph = ${safeJson(graph)};
  var nodes = new vis.DataSet(graph.nodes.map(function(n){
    return Object.assign({}, n, {
      shape: 'box',
      shapeProperties: { borderRadius: 8 },
      margin: { top: 8, right: 12, bottom: 8, left: 12 },
      font: { face: 'sans-serif', size: 13 }
    });
  }));
  var edges = new vis.DataSet(graph.edges.map(function(e, i){
    return Object.assign({ id: 'e' + i }, e, {
      color: e.color ? { color: e.color } : { color: '#C5CEDB' },
      arrows: 'to',
      font: { size: 10, color: '#8C9BB5', strokeWidth: 0 }
    });
  }));
  var network = new vis.Network(
    document.getElementById('net'),
    { nodes: nodes, edges: edges },
    {
      groups: ${safeJson(groups)},
      nodes: { borderWidth: 0 },
      edges: { smooth: { enabled: true, type: 'dynamic', roundness: 0.5 }, width: 1.2 },
      interaction: { hover: false, dragNodes: true, zoomView: true, dragView: true },
      physics: {
        enabled: true,
        stabilization: true,
        barnesHut: { gravitationalConstant: -2600, springLength: 110, springConstant: 0.04, avoidOverlap: 0.2 }
      }
    }
  );
  network.on('click', function(params){
    if (params.nodes && params.nodes.length) {
      var id = String(params.nodes[0]);
      if (id.indexOf('drug:') === 0 && window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify({ type: 'drug', drug: id.slice(5) }));
      }
    }
  });
})();
</script>
</body>
</html>`;
}

export default function GraphWebView({ graph, onDrugTap }: Props) {
  const [visJs, setVisJs] = useState<string | null>(visJsCache);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    if (!visJs) {
      loadVisJs()
        .then((js) => {
          if (active) setVisJs(js);
        })
        .catch((e: unknown) => {
          if (active) setError(e instanceof Error ? e.message : String(e));
        });
    }
    return () => {
      active = false;
    };
  }, [visJs]);

  const html = useMemo(
    () => (visJs ? buildHtml(visJs, graph) : null),
    [visJs, graph],
  );

  const onMessage = (event: WebViewMessageEvent) => {
    try {
      const msg = JSON.parse(event.nativeEvent.data) as {
        type?: string;
        drug?: string;
      };
      if (msg.type === 'drug' && msg.drug && onDrugTap) onDrugTap(msg.drug);
    } catch {
      // ignore malformed messages
    }
  };

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Graph failed to load: {error}</Text>
      </View>
    );
  }

  if (!html) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.loadingText}>Loading graph engine…</Text>
      </View>
    );
  }

  return (
    <WebView
      originWhitelist={['*']}
      source={{ html }}
      javaScriptEnabled
      scrollEnabled={false}
      bounces={false}
      style={styles.webview}
      onMessage={onMessage}
      setBuiltInZoomControls={false}
    />
  );
}

const styles = StyleSheet.create({
  webview: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.bg,
  },
  loadingText: {
    fontFamily: fonts.body,
    fontSize: 13,
    color: colors.textMuted,
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 13,
    color: colors.red,
    paddingHorizontal: spacing.xl,
    textAlign: 'center',
  },
});

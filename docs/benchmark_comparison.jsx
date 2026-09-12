import { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, LabelList
} from "recharts";

const RAW = [
  { Method: "fastsae", Min: 0.000499503046739846, Median: 0.00058806297602132, IPS: 1622.53773721854, Mem: 0.00506591796875, n: 30, Algo: "EBLUP" },
  { Method: "emdi", Min: 0.0106049370369874, Median: 0.0108802110189572, IPS: 91.1934476514222, Mem: 4.18307495117188, n: 30, Algo: "EBLUP" },
  { Method: "sae", Min: 0.00151544203981757, Median: 0.00155166554031894, IPS: 631.905967674181, Mem: 0.15338134765625, n: 30, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.000498806010000408, Median: 0.000510982994455844, IPS: 1912.06792148805, Mem: 0.00884246826171875, n: 50, Algo: "EBLUP" },
  { Method: "emdi", Min: 0.0167840060312301, Median: 0.0180643130443059, IPS: 51.9488538005359, Mem: 10.4675903320312, n: 50, Algo: "EBLUP" },
  { Method: "sae", Min: 0.00187525805085897, Median: 0.00195069803157821, IPS: 494.429233310148, Mem: 0.373138427734375, n: 50, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.000530130055267364, Median: 0.000550220021978021, IPS: 1761.40355199163, Mem: 0.0162811279296875, n: 100, Algo: "EBLUP" },
  { Method: "emdi", Min: 0.0495857690111734, Median: 0.0505075720138848, IPS: 19.4235557225948, Mem: 37.3394088745117, n: 100, Algo: "EBLUP" },
  { Method: "sae", Min: 0.00336212298134342, Median: 0.00344266754109412, IPS: 288.108339998426, Mem: 0.91754150390625, n: 100, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.000872233998961747, Median: 0.000899663020391017, IPS: 1080.76611815701, Mem: 0.0385971069335938, n: 250, Algo: "EBLUP" },
  { Method: "emdi", Min: 0.839817391010001, Median: 0.857347269018646, IPS: 1.16288651206072, Mem: 240.89331817627, n: 250, Algo: "EBLUP" },
  { Method: "sae", Min: 0.0372851950232871, Median: 0.0375750035163946, IPS: 26.0152750354733, Mem: 6.34988403320312, n: 250, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.00138416001573205, Median: 0.00140398350777104, IPS: 696.95570024357, Mem: 0.0757904052734375, n: 500, Algo: "EBLUP" },
  { Method: "emdi", Min: 5.82063773396658, Median: 5.8636555080011, IPS: 0.169882925123163, Mem: 880.313690185547, n: 500, Algo: "EBLUP" },
  { Method: "sae", Min: 0.196524069993757, Median: 0.196725974499714, IPS: 5.02801718604854, Mem: 18.2390747070312, n: 500, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.00395563902566209, Median: 0.00413357905927114, IPS: 175.754886710274, Mem: 0.185699462890625, n: 1000, Algo: "EBLUP" },
  { Method: "emdi", Min: 50.6428132470464, Median: 51.022360681498, IPS: 0.0195765812738249, Mem: 3773.59893798828, n: 1000, Algo: "EBLUP" },
  { Method: "sae", Min: 1.49632952199318, Median: 1.50363106853911, IPS: 0.664077313860355, Mem: 71.9765243530273, n: 1000, Algo: "EBLUP" },
  { Method: "fastsae", Min: 0.000720124051440507, Median: 0.000734617526177317, IPS: 1312.17464758992, Mem: 0.0298080444335938, n: 30, Algo: "SEBLUP" },
  { Method: "emdi", Min: 0.0105219940305687, Median: 0.0107104709895793, IPS: 93.4634624983975, Mem: 4.18307495117188, n: 30, Algo: "SEBLUP" },
  { Method: "sae", Min: 0.00388958799885586, Median: 0.00410110701341182, IPS: 246.219493111412, Mem: 1.37313842773438, n: 30, Algo: "SEBLUP" },
  { Method: "fastsae", Min: 0.00110367900924757, Median: 0.00114812300307676, IPS: 841.216922087629, Mem: 0.0737838745117188, n: 50, Algo: "SEBLUP" },
  { Method: "emdi", Min: 0.0172649360029027, Median: 0.0180004760040902, IPS: 43.8397564439812, Mem: 10.4675903320312, n: 50, Algo: "SEBLUP" },
  { Method: "sae", Min: 0.00938670395407826, Median: 0.00967977198888548, IPS: 102.570649239143, Mem: 3.31219482421875, n: 50, Algo: "SEBLUP" },
  { Method: "fastsae", Min: 0.00462455401429906, Median: 0.0047257625265047, IPS: 208.133010809676, Mem: 0.25860595703125, n: 100, Algo: "SEBLUP" },
  { Method: "emdi", Min: 0.0497629710589536, Median: 0.052989999006968, IPS: 17.4720875710885, Mem: 37.3394088745117, n: 100, Algo: "SEBLUP" },
  { Method: "sae", Min: 0.0698495680117048, Median: 0.0705356210237369, IPS: 12.7922295608982, Mem: 17.9755172729492, n: 100, Algo: "SEBLUP" },
  { Method: "fastsae", Min: 0.0235872999764979, Median: 0.0238317214534618, IPS: 37.0116185457864, Mem: 1.49971771240234, n: 250, Algo: "SEBLUP" },
  { Method: "emdi", Min: 0.839309483009856, Median: 0.843781660543755, IPS: 1.17485877939886, Mem: 240.89331817627, n: 250, Algo: "SEBLUP" },
  { Method: "sae", Min: 1.05539297201904, Median: 1.07476272500935, IPS: 0.925568177655969, Mem: 110.387939453125, n: 250, Algo: "SEBLUP" },
  { Method: "fastsae", Min: 0.823366059048567, Median: 0.832157648022985, IPS: 1.19407778305699, Mem: 23.1548004150391, n: 1000, Algo: "SEBLUP" },
  { Method: "emdi", Min: 50.8262589730439, Median: 51.0593093895295, IPS: 0.0195518056960103, Mem: 3772.56869506836, n: 1000, Algo: "SEBLUP" },
  { Method: "sae", Min: 66.9846985759796, Median: 67.1517531580175, IPS: 0.0148793969215074, Mem: 1920.00576019287, n: 1000, Algo: "SEBLUP" },
  { Method: "fastsae", Min: 0.133096906007268, Median: 0.142050649999874, IPS: 6.95660261337659, Mem: 5.8570556640625, n: 500, Algo: "SEBLUP" },
  { Method: "emdi", Min: 3.08698893297696, Median: 3.13661592750577, IPS: 0.31767670164998, Mem: 949.864486694336, n: 500, Algo: "SEBLUP" },
  { Method: "sae", Min: 7.10022227297304, Median: 7.11792668796261, IPS: 0.139645160966639, Mem: 776.406730651855, n: 500, Algo: "SEBLUP" },
];

const N_LEVELS = [30, 50, 100, 250, 500, 1000];
const METHODS = ["fastsae", "sae", "emdi"];
const COLORS = { fastsae: "#1D6A5C", sae: "#C77B3B", emdi: "#7A5FA8" };

function pivot(algo, key) {
  return N_LEVELS.map((n) => {
    const row = { n: String(n) };
    METHODS.forEach((m) => {
      const rec = RAW.find((r) => r.Algo === algo && r.n === n && r.Method === m);
      row[m] = rec ? rec[key] : null;
    });
    return row;
  });
}

function speedup(algo) {
  return N_LEVELS.map((n) => {
    const f = RAW.find((r) => r.Algo === algo && r.n === n && r.Method === "fastsae");
    const s = RAW.find((r) => r.Algo === algo && r.n === n && r.Method === "sae");
    const e = RAW.find((r) => r.Algo === algo && r.n === n && r.Method === "emdi");
    return {
      n: String(n),
      vsSae: f && s ? s.Median / f.Median : null,
      vsEmdi: f && e ? e.Median / f.Median : null,
    };
  });
}

const METRICS = {
  time: { key: "Median", label: "Waktu eksekusi (median, detik)", fmt: (v) => (v < 0.001 ? `${(v * 1e6).toFixed(0)} µs` : v < 1 ? `${(v * 1000).toFixed(2)} ms` : `${v.toFixed(2)} s`) },
  mem: { key: "Mem", label: "Memori terpakai (MB)", fmt: (v) => `${v.toFixed(2)} MB` },
  ips: { key: "IPS", label: "Iterasi per detik", fmt: (v) => v.toFixed(2) },
};

function CustomTooltip({ active, payload, label, fmt }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{ background: "#fff", border: "1px solid #E4DFD6", borderRadius: 6, padding: "8px 12px", fontSize: 13 }}>
      <div style={{ fontWeight: 600, marginBottom: 4, color: "#3A3530" }}>n = {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey}: {fmt ? fmt(p.value) : p.value}
        </div>
      ))}
    </div>
  );
}

export default function BenchmarkDashboard() {
  const [algo, setAlgo] = useState("EBLUP");
  const [metric, setMetric] = useState("time");

  const chartData = useMemo(() => pivot(algo, METRICS[metric].key), [algo, metric]);
  const speedupData = useMemo(() => speedup(algo), [algo]);

  return (
    <div style={{ fontFamily: "'Georgia', 'Iowan Old Style', serif", background: "#FBF9F4", minHeight: "100%", padding: "28px 20px", color: "#2E2A24" }}>
      <div style={{ maxWidth: 880, margin: "0 auto" }}>
        <div style={{ marginBottom: 4, fontSize: 12, letterSpacing: 0.4, color: "#8A8175", fontFamily: "'Helvetica Neue', Arial, sans-serif" }}>
          fastsae &middot; sae &middot; emdi
        </div>
        <h1 style={{ fontSize: 30, fontWeight: 400, margin: "0 0 6px", lineHeight: 1.15 }}>
          Seberapa cepat fastsae, dibanding sae dan emdi?
        </h1>
        <p style={{ fontSize: 14.5, color: "#5E574C", margin: "0 0 24px", maxWidth: 620, fontFamily: "'Helvetica Neue', Arial, sans-serif", lineHeight: 1.5 }}>
          Perbandingan waktu eksekusi, penggunaan memori, dan throughput untuk algoritma EBLUP dan SEBLUP, pada ukuran data (n) dari 30 hingga 1000 area.
        </p>

        {/* Controls */}
        <div style={{ display: "flex", gap: 24, marginBottom: 20, flexWrap: "wrap", fontFamily: "'Helvetica Neue', Arial, sans-serif" }}>
          <div style={{ display: "flex", gap: 6 }}>
            {["EBLUP", "SEBLUP"].map((a) => (
              <button
                key={a}
                onClick={() => setAlgo(a)}
                style={{
                  padding: "6px 16px",
                  borderRadius: 20,
                  border: `1px solid ${algo === a ? "#2E2A24" : "#D8D2C6"}`,
                  background: algo === a ? "#2E2A24" : "transparent",
                  color: algo === a ? "#FBF9F4" : "#5E574C",
                  fontSize: 13,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                {a}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {Object.entries({ time: "Waktu", mem: "Memori", ips: "Throughput" }).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setMetric(k)}
                style={{
                  padding: "6px 16px",
                  borderRadius: 20,
                  border: `1px solid ${metric === k ? "#1D6A5C" : "#D8D2C6"}`,
                  background: metric === k ? "#EAF2EF" : "transparent",
                  color: metric === k ? "#1D6A5C" : "#5E574C",
                  fontSize: 13,
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Main line chart */}
        <div style={{ background: "#fff", border: "1px solid #E9E4D9", borderRadius: 10, padding: "20px 16px 8px" }}>
          <div style={{ fontSize: 13, color: "#8A8175", marginBottom: 8, fontFamily: "'Helvetica Neue', Arial, sans-serif" }}>
            {METRICS[metric].label} &mdash; {algo} &middot; skala logaritmik
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 5 }}>
              <CartesianGrid stroke="#F0ECE3" vertical={false} />
              <XAxis dataKey="n" tick={{ fontSize: 12, fill: "#8A8175" }} label={{ value: "n (jumlah area)", position: "insideBottom", offset: -3, fontSize: 12, fill: "#8A8175" }} />
              <YAxis
                scale="log"
                domain={["auto", "auto"]}
                tick={{ fontSize: 11, fill: "#8A8175" }}
                tickFormatter={(v) => METRICS[metric].fmt(v)}
                width={70}
              />
              <Tooltip content={<CustomTooltip fmt={METRICS[metric].fmt} />} />
              <Legend wrapperStyle={{ fontSize: 12, fontFamily: "'Helvetica Neue', Arial, sans-serif" }} />
              {METHODS.map((m) => (
                <Line key={m} type="monotone" dataKey={m} stroke={COLORS[m]} strokeWidth={2.4} dot={{ r: 3.5 }} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Speedup bar chart */}
        <div style={{ background: "#fff", border: "1px solid #E9E4D9", borderRadius: 10, padding: "20px 16px 8px", marginTop: 18 }}>
          <div style={{ fontSize: 13, color: "#8A8175", marginBottom: 8, fontFamily: "'Helvetica Neue', Arial, sans-serif" }}>
            Faktor percepatan fastsae dibanding kompetitor &mdash; {algo}
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={speedupData} margin={{ top: 5, right: 20, bottom: 5, left: 5 }}>
              <CartesianGrid stroke="#F0ECE3" vertical={false} />
              <XAxis dataKey="n" tick={{ fontSize: 12, fill: "#8A8175" }} label={{ value: "n (jumlah area)", position: "insideBottom", offset: -3, fontSize: 12, fill: "#8A8175" }} />
              <YAxis
                scale="log"
                domain={["auto", "auto"]}
                tick={{ fontSize: 11, fill: "#8A8175" }}
                tickFormatter={(v) => `${v.toFixed(0)}x`}
                width={50}
              />
              <Tooltip formatter={(v) => `${v.toFixed(1)}x lebih cepat`} labelFormatter={(l) => `n = ${l}`} contentStyle={{ fontSize: 13, borderRadius: 6, border: "1px solid #E4DFD6" }} />
              <Legend wrapperStyle={{ fontSize: 12, fontFamily: "'Helvetica Neue', Arial, sans-serif" }} formatter={(v) => (v === "vsSae" ? "vs sae" : "vs emdi")} />
              <Bar dataKey="vsSae" fill={COLORS.sae} radius={[3, 3, 0, 0]}>
                <LabelList dataKey="vsSae" position="top" formatter={(v) => `${v.toFixed(0)}x`} fontSize={10} fill="#8A8175" />
              </Bar>
              <Bar dataKey="vsEmdi" fill={COLORS.emdi} radius={[3, 3, 0, 0]}>
                <LabelList dataKey="vsEmdi" position="top" formatter={(v) => `${v.toFixed(0)}x`} fontSize={10} fill="#8A8175" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ marginTop: 16, fontSize: 12, color: "#A69E90", fontFamily: "'Helvetica Neue', Arial, sans-serif", lineHeight: 1.6 }}>
          Sumbu waktu, memori, dan throughput ditampilkan dalam skala logaritmik karena rentang nilai antar package sangat lebar (hingga 4 orde besaran). Faktor percepatan dihitung sebagai median waktu kompetitor dibagi median waktu fastsae pada n dan algoritma yang sama.
        </div>
      </div>
    </div>
  );
}

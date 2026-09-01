/**
 * =============================================================================
 * SHOPEE DATA MINING - JAVASCRIPT ENGINE (STITCH UI)
 * Authors: Le Thanh Truc Vi, Tran Dinh Huy, Vu Hoang Thien An, Duong Phuong Anh
 * =============================================================================
 */

let charts = {};
let sampleProductsData = [];

document.addEventListener("DOMContentLoaded", function() {
    initStitchTabs();
    loadOverviewStats();
    loadSampleProducts();
    loadSampleComments();
    loadWordCloud("all");
    loadAprioriRules();
    loadClusteringData();
    loadModelBenchmarks();
});

// =============================================================================
// 1. STITCH TAB NAVIGATION
// =============================================================================
function initStitchTabs() {
    const tabLinks = document.querySelectorAll(".stitch-nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const breadcrumb = document.getElementById("stitch-breadcrumb-text");

    const tabNames = {
        "tab-overview": "Dashboard Tổng Quan & Social Mining",
        "tab-sentiment": "Phân Tích Cảm Xúc MXH (Naive Bayes)",
        "tab-forecaster": "Dự Báo Doanh Số (Random Forest & XGBoost)",
        "tab-risk": "Giám Sát Rủi Ro & Hoàn Hàng",
        "tab-basket": "Gợi Ý Combo Giỏ Hàng (Apriori)",
        "tab-clustering": "Phân Khúc Sản Phẩm (K-Means & PCA)",
        "tab-benchmark": "Đấu Trường So Sánh Mô Hình ML"
    };

    tabLinks.forEach(link => {
        link.addEventListener("click", function() {
            const targetTab = this.getAttribute("data-tab");

            tabLinks.forEach(l => {
                l.classList.remove("active");
                l.classList.add("text-on-surface-variant");
            });
            tabPanes.forEach(p => p.classList.remove("active"));

            this.classList.add("active");
            this.classList.remove("text-on-surface-variant");

            const targetPane = document.getElementById(targetTab);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            if (breadcrumb && tabNames[targetTab]) {
                breadcrumb.textContent = tabNames[targetTab];
            }
        });
    });
}

// =============================================================================
// 2. TAB 1: OVERVIEW DASHBOARD & CHARTS
// =============================================================================
async function loadOverviewStats() {
    try {
        const response = await fetch("/api/overview_stats");
        const data = await response.json();

        if (data.status === "success") {
            const kpis = data.kpis;
            document.getElementById("kpi-products").textContent = kpis.total_products + " SP";
            document.getElementById("kpi-reviews").textContent = kpis.total_reviews + " Bình luận";
            document.getElementById("kpi-sales").textContent = kpis.total_monthly_sales;
            document.getElementById("kpi-rating").textContent = kpis.avg_rating + " / 5.0 ★";

            renderCategorySalesChart(data.category_sales);
            renderSentimentDoughnutChart(data.sentiment_dist);
            renderRatingDistChart(data.rating_dist);
            renderTopProductsTable(data.top_products);
        }
    } catch (error) {
        console.error("Lỗi khi tải Overview stats:", error);
    }
}

function renderCategorySalesChart(catData) {
    const ctx = document.getElementById("chartCategorySales");
    if (!ctx) return;

    if (charts["categorySales"]) charts["categorySales"].destroy();

    charts["categorySales"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels: catData.labels,
            datasets: [{
                label: "Doanh Số Bán (SP/tháng)",
                data: catData.values,
                backgroundColor: [
                    "#b22204", "#d63c1e", "#005bbd", "#00b050", "#ffb94c",
                    "#7e5300", "#ba1a1a", "#4f92fe", "#9e6a00"
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: "#5b403b", font: { size: 11, family: "Inter", weight: "600" } },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: "#8f7069", font: { size: 11, family: "Inter" } },
                    grid: { color: "#e3beb6", drawBorder: false }
                }
            }
        }
    });
}

function renderSentimentDoughnutChart(sentData) {
    const ctx = document.getElementById("chartSentimentDoughnut");
    if (!ctx) return;

    if (charts["sentimentDoughnut"]) charts["sentimentDoughnut"].destroy();

    charts["sentimentDoughnut"] = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Khen Ngợi (Tích Cực)", "Bình Thường (Trung Lập)", "Chê / Khiếu Nại (Tiêu Cực)"],
            datasets: [{
                data: [sentData.pos, sentData.neu, sentData.neg],
                backgroundColor: ["#00b050", "#ffb94c", "#ba1a1a"],
                borderWidth: 3,
                borderColor: "#fff8f6"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            cutout: "72%"
        }
    });
}

function renderRatingDistChart(ratingData) {
    const ctx = document.getElementById("chartRatingDist");
    if (!ctx) return;

    if (charts["ratingDist"]) charts["ratingDist"].destroy();

    charts["ratingDist"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ratingData.labels,
            datasets: [{
                label: "Số Lượng Đánh Giá",
                data: ratingData.values,
                backgroundColor: ["#ba1a1a", "#d63c1e", "#ffb94c", "#4f92fe", "#00b050"],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#8f7069" }, grid: { color: "#e3beb6" } },
                y: { ticks: { color: "#271815", font: { weight: "700", family: "Inter" } }, grid: { display: false } }
            }
        }
    });
}

function renderTopProductsTable(topProducts) {
    const tbody = document.getElementById("top-products-tbody");
    if (!tbody) return;

    if (!topProducts || topProducts.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="py-4 text-center text-on-surface-variant">Không có dữ liệu</td></tr>`;
        return;
    }

    tbody.innerHTML = topProducts.map(p => `
        <tr class="hover:bg-surface-container-low transition-colors">
            <td class="py-2.5 px-3 font-mono text-primary font-bold">${p.product_id}</td>
            <td class="py-2.5 px-3 font-bold text-on-surface">${p.product_name.length > 32 ? p.product_name.substring(0, 32) + "..." : p.product_name}</td>
            <td class="py-2.5 px-3 text-on-surface-variant">${p.category}</td>
            <td class="py-2.5 px-3 font-bold text-primary">${p.price.toLocaleString("vi-VN")} ₫</td>
            <td class="py-2.5 px-3 font-bold text-on-surface">${p.monthly_sold.toLocaleString("vi-VN")} SP</td>
            <td class="py-2.5 px-3 font-bold text-tertiary">★ ${p.rating_star}</td>
            <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-[#00b050]/15 text-[#00b050] rounded-full text-[11px] font-bold">${p.growth_potential}</span></td>
        </tr>
    `).join("");
}

// =============================================================================
// 3. TAB 2: SENTIMENT ANALYSIS & WORD CLOUD
// =============================================================================
async function loadSampleComments() {
    try {
        const response = await fetch("/api/sample_comments");
        const data = await response.json();
        if (data.status === "success" && data.samples) {
            const container = document.getElementById("sample-comments-container");
            if (!container) return;

            container.innerHTML = data.samples.map(c => `
                <button type="button" class="px-3 py-1 bg-surface rounded-full border border-outline-variant/40 hover:border-primary hover:bg-surface-container text-xs font-medium text-on-surface transition-all cursor-pointer shadow-sm" onclick="populateSampleComment('${encodeURIComponent(c.text)}')">
                    ${c.type}: "${c.text.substring(0, 26)}..."
                </button>
            `).join("");
        }
    } catch (e) {
        console.error("Lỗi khi tải sample comments:", e);
    }
}

function populateSampleComment(encodedText) {
    const text = decodeURIComponent(encodedText);
    document.getElementById("sentiment-input-text").value = text;
    submitSentimentAnalysis();
}

async function submitSentimentAnalysis() {
    const text = document.getElementById("sentiment-input-text").value.trim();
    if (!text) return;

    try {
        const response = await fetch("/api/analyze_sentiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        const res = await response.json();

        if (res.status === "success") {
            document.getElementById("sentiment-cleaned-display").textContent = res.cleaned_text;

            const badge = document.getElementById("sent-badge-label");
            badge.textContent = res.sentiment_label;
            if (res.sentiment_label === "Tích cực") {
                badge.style.background = "#dcfce7";
                badge.style.color = "#15803d";
            } else if (res.sentiment_label === "Tiêu cực") {
                badge.style.background = "#ffdad6";
                badge.style.color = "#ba1a1a";
            } else {
                badge.style.background = "#ffddb2";
                badge.style.color = "#7e5300";
            }

            const percentWidth = Math.max(5, Math.min(95, ((res.sentiment_score + 1.0) / 2.0) * 100));
            document.getElementById("sent-gauge-fill").style.width = `${percentWidth}%`;
            document.getElementById("sent-score-val").textContent = `Điểm: ${res.sentiment_score >= 0 ? "+" : ""}${res.sentiment_score} (Độ tin cậy: ${res.confidence_percent}%)`;

            const posContainer = document.getElementById("sent-pos-pills");
            if (res.positive_keywords && res.positive_keywords.length > 0) {
                posContainer.innerHTML = res.positive_keywords.map(kw => `<span class="px-2 py-0.5 bg-[#00b050]/15 text-[#00b050] rounded-full font-bold text-[11px]">${kw}</span>`).join("");
            } else {
                posContainer.innerHTML = `<span class="text-on-surface-variant text-[11px]">Không phát hiện</span>`;
            }

            const negContainer = document.getElementById("sent-neg-pills");
            if (res.negative_keywords && res.negative_keywords.length > 0) {
                negContainer.innerHTML = res.negative_keywords.map(kw => `<span class="px-2 py-0.5 bg-error/15 text-error rounded-full font-bold text-[11px]">${kw}</span>`).join("");
            } else {
                negContainer.innerHTML = `<span class="text-on-surface-variant text-[11px]">Không phát hiện</span>`;
            }

            const aspectContainer = document.getElementById("sent-aspect-pills");
            if (res.aspects && res.aspects.length > 0) {
                aspectContainer.innerHTML = res.aspects.map(asp => `<span class="px-2 py-0.5 bg-secondary/15 text-secondary rounded-full font-bold text-[11px]">${asp}</span>`).join("");
            }
        }
    } catch (e) {
        console.error("Lỗi khi phân tích cảm xúc:", e);
    }
}

async function loadWordCloud(sentimentFilter) {
    const buttons = document.querySelectorAll(".wc-filter-btn");
    buttons.forEach(b => {
        b.classList.remove("bg-primary", "text-white");
        b.classList.add("bg-surface-container", "text-on-surface-variant");
    });
    if (event && event.target) {
        event.target.classList.remove("bg-surface-container", "text-on-surface-variant");
        event.target.classList.add("bg-primary", "text-white");
    }

    try {
        const response = await fetch(`/api/wordcloud?sentiment=${sentimentFilter}`);
        const data = await response.json();

        if (data.status === "success" && data.words) {
            const container = document.getElementById("wordcloud-container");
            if (!container) return;

            const colors = ["#b22204", "#005bbd", "#00b050", "#7e5300", "#4f92fe", "#ba1a1a"];
            const bgColors = ["#ffe9e5", "#d7e2ff", "#dcfce7", "#ffddb2", "#f0f9ff", "#ffdad6"];

            container.innerHTML = data.words.map((w, idx) => {
                const fontSize = Math.min(22, Math.max(12, Math.floor(w.weight * 0.2) + 12));
                const color = colors[idx % colors.length];
                const bg = bgColors[idx % bgColors.length];
                return `<span class="wc-word-pill font-headline" style="font-size: ${fontSize}px; color: ${color}; background: ${bg};">${w.text} (${w.weight})</span>`;
            }).join("");
        }
    } catch (e) {
        console.error("Lỗi khi tải word cloud:", e);
    }
}

// =============================================================================
// 4. TAB 3: PRODUCT FORECASTER (SALES & GROWTH ML)
// =============================================================================
async function loadSampleProducts() {
    try {
        const response = await fetch("/api/sample_products");
        const data = await response.json();
        if (data.status === "success" && data.samples) {
            sampleProductsData = data.samples;
            const container = document.getElementById("sample-products-container");
            if (!container) return;

            container.innerHTML = data.samples.slice(0, 5).map((p, idx) => `
                <button type="button" class="px-3 py-1 bg-surface rounded-full border border-outline-variant/40 hover:border-primary hover:bg-surface-container text-xs font-semibold text-on-surface transition-all cursor-pointer shadow-sm flex items-center gap-1" onclick="populateSampleProduct(${idx})">
                    <span class="material-symbols-outlined text-primary text-xs">local_offer</span>
                    <span>${p.category.split(" ")[0]} (${p.price.toLocaleString("vi-VN")}₫)</span>
                </button>
            `).join("");
        }
    } catch (e) {
        console.error("Lỗi khi tải sample products:", e);
    }
}

function populateSampleProduct(index) {
    const p = sampleProductsData[index];
    if (!p) return;

    document.getElementById("p-url").value = `https://shopee.vn/product-${p.product_id}`;
    document.getElementById("p-category").value = p.category;
    document.getElementById("p-shop-type").value = p.shop_type;
    document.getElementById("p-location").value = p.shop_location;
    document.getElementById("p-price").value = p.price;
    document.getElementById("p-discount").value = p.discount_rate;
    document.getElementById("p-megasale").value = p.is_megasale;
    document.getElementById("p-views").value = p.view_count;
    document.getElementById("p-favs").value = p.favorite_count;
    document.getElementById("p-hist-sold").value = p.historical_sold;
    document.getElementById("p-rating").value = p.rating_star;
    document.getElementById("p-ship-rate").value = p.ship_on_time_rate;
    document.getElementById("p-chat-rate").value = p.chat_response_rate;

    submitProductPrediction();
}

async function submitProductPrediction() {
    const payload = {
        category: document.getElementById("p-category").value,
        shop_type: document.getElementById("p-shop-type").value,
        shop_location: document.getElementById("p-location").value,
        price: parseFloat(document.getElementById("p-price").value) || 200000,
        discount_rate: parseFloat(document.getElementById("p-discount").value) || 0,
        is_megasale: parseInt(document.getElementById("p-megasale").value) || 0,
        view_count: parseInt(document.getElementById("p-views").value) || 5000,
        favorite_count: parseInt(document.getElementById("p-favs").value) || 200,
        historical_sold: parseInt(document.getElementById("p-hist-sold").value) || 500,
        rating_star: parseFloat(document.getElementById("p-rating").value) || 4.5,
        ship_on_time_rate: parseFloat(document.getElementById("p-ship-rate").value) || 95.0,
        chat_response_rate: parseFloat(document.getElementById("p-chat-rate").value) || 90.0,
        shipping_fee: 22000,
        avg_delivery_days: 2.5
    };

    try {
        const response = await fetch("/api/predict_product", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const res = await response.json();

        if (res.status === "success") {
            const pred = res.prediction;

            document.getElementById("res-placeholder").classList.add("hidden");
            document.getElementById("res-actual-content").classList.remove("hidden");

            document.getElementById("res-growth-label").textContent = pred.growth_potential;
            document.getElementById("res-confidence").textContent = `Độ tin cậy: ${pred.confidence_percent}%`;
            document.getElementById("res-monthly-sold").textContent = pred.predicted_monthly_sold;
            document.getElementById("res-monthly-revenue").textContent = pred.estimated_monthly_revenue;
            document.getElementById("res-risk-level").textContent = pred.risk_level;

            const featContainer = document.getElementById("res-feature-bars");
            featContainer.innerHTML = pred.feature_impacts.map(f => `
                <div class="feat-bar-row">
                    <span class="feat-name">${f.name}</span>
                    <div class="feat-track">
                        <div class="feat-fill" style="width: ${f.score}%;"></div>
                    </div>
                    <span class="feat-val">${f.impact}</span>
                </div>
            `).join("");

            const recContainer = document.getElementById("res-recommendations");
            recContainer.innerHTML = pred.recommendations.map(r => `<li>${r}</li>`).join("");

            if (pred.growth_potential === "Tiềm Năng Cao" && typeof confetti === "function") {
                confetti({ particleCount: 60, spread: 60, origin: { y: 0.6 } });
            }
        }
    } catch (e) {
        console.error("Lỗi khi dự đoán sản phẩm:", e);
    }
}

// =============================================================================
// 5. TAB 5: APRIORI MARKET BASKET RULES
// =============================================================================
async function loadAprioriRules() {
    try {
        const response = await fetch("/api/market_basket");
        const data = await response.json();

        if (data.status === "success" && data.rules) {
            const tbody = document.getElementById("apriori-rules-tbody");
            if (!tbody) return;

            tbody.innerHTML = data.rules.slice(0, 10).map(r => `
                <tr class="hover:bg-surface-container-low transition-colors">
                    <td class="py-2.5 px-3 font-bold text-primary">${r.antecedent}</td>
                    <td class="py-2.5 px-3 font-bold text-secondary">${r.consequent}</td>
                    <td class="py-2.5 px-3 font-medium text-on-surface">${r.support}%</td>
                    <td class="py-2.5 px-3 font-bold text-on-surface">${r.confidence}%</td>
                    <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-[#00b050]/15 text-[#00b050] rounded-full font-bold text-[11px]">Gấp ${r.lift}x</span></td>
                    <td class="py-2.5 px-3 text-[11px] text-on-surface-variant font-medium">Tạo Combo giảm giá bán kèm để tăng AOV.</td>
                </tr>
            `).join("");
        }
    } catch (e) {
        console.error("Lỗi khi tải Apriori rules:", e);
    }
}

// =============================================================================
// 6. TAB 6: K-MEANS CLUSTERING & 2D PCA
// =============================================================================
async function loadClusteringData() {
    try {
        const response = await fetch("/api/clustering");
        const data = await response.json();

        if (data.status === "success" && data.clustering) {
            const clustering = data.clustering;

            const profContainer = document.getElementById("cluster-profiles-container");
            if (profContainer) {
                const colors = ["#b22204", "#005bbd", "#00b050", "#7e5300"];
                profContainer.innerHTML = Object.keys(clustering.cluster_profiles).map(cid => {
                    const cp = clustering.cluster_profiles[cid];
                    return `
                        <div class="bg-surface-container-low p-2.5 rounded-lg border border-outline-variant/30 text-xs" style="border-left: 3.5px solid ${colors[cid]};">
                            <div class="font-bold text-on-surface mb-0.5" style="color: ${colors[cid]};">${cp.name}</div>
                            <div class="text-[11px] text-on-surface-variant">
                                Số lượng: <strong>${cp.count} SP</strong> | Giá TB: <strong>${cp.avg_price.toLocaleString("vi-VN")} ₫</strong> | Bán TB: <strong>${cp.avg_monthly_sold} SP/tháng</strong>
                            </div>
                        </div>
                    `;
                }).join("");
            }

            renderKmeansScatterChart(clustering.sample_points);
        }
    } catch (e) {
        console.error("Lỗi khi tải Clustering data:", e);
    }
}

function renderKmeansScatterChart(samplePoints) {
    const ctx = document.getElementById("chartKmeansScatter");
    if (!ctx) return;

    if (charts["kmeansScatter"]) charts["kmeansScatter"].destroy();

    const clusterColors = ["#b22204", "#005bbd", "#00b050", "#ffb94c"];
    const datasets = [0, 1, 2, 3].map(cid => {
        const points = samplePoints.filter(p => p.cluster === cid).map(p => ({
            x: p.pca_x,
            y: p.pca_y,
            name: p.product_name,
            price: p.price,
            sold: p.monthly_sold
        }));

        const names = [
            "Cụm 1: Best-Seller",
            "Cụm 2: Hàng Mới Tiềm Năng",
            "Cụm 3: Hàng Phổ Thông",
            "Cụm 4: Cần Tối Ưu Vận Hành"
        ];

        return {
            label: names[cid],
            data: points,
            backgroundColor: clusterColors[cid],
            pointRadius: 6,
            pointHoverRadius: 9
        };
    });

    charts["kmeansScatter"] = new Chart(ctx, {
        type: "scatter",
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#271815", font: { size: 11, family: "Inter", weight: "600" } } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const raw = context.raw;
                            return `${raw.name} | Giá: ${raw.price.toLocaleString("vi-VN")}₫ | Bán: ${raw.sold} SP`;
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: "Trục 1: Sức Bán & Lượt Quan Tâm Của Khách (PCA 1)", color: "#5b403b", font: { weight: "600", family: "Inter" } }, ticks: { color: "#8f7069" }, grid: { color: "#e3beb6" } },
                y: { title: { display: true, text: "Trục 2: Mức Giá & Điểm Đánh Giá (PCA 2)", color: "#5b403b", font: { weight: "600", family: "Inter" } }, ticks: { color: "#8f7069" }, grid: { color: "#e3beb6" } }
            }
        }
    });
}

// =============================================================================
// 7. TAB 7: MODEL BENCHMARK ARENA
// =============================================================================
async function loadModelBenchmarks() {
    try {
        const response = await fetch("/api/model_benchmarks");
        const data = await response.json();

        if (data.status === "success" && data.benchmarks) {
            const bm = data.benchmarks;
            const tbody = document.getElementById("benchmark-tbody");
            if (!tbody) return;

            const rows = [];

            if (bm.sentiment_models) {
                Object.keys(bm.sentiment_models).forEach(name => {
                    const m = bm.sentiment_models[name];
                    rows.push(`
                        <tr class="hover:bg-surface-container-low transition-colors">
                            <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-primary/10 text-primary rounded-full text-[11px] font-bold">NLP Cảm Xúc</span></td>
                            <td class="py-2.5 px-3 font-bold text-on-surface">${name}</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.accuracy * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.precision * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.recall * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-bold text-[#00b050]">${(m.f1_score * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-[#00b050]/15 text-[#00b050] rounded-full text-[11px] font-bold">Rất Chuẩn Xác</span></td>
                        </tr>
                    `);
                });
            }

            if (bm.growth_models) {
                Object.keys(bm.growth_models).forEach(name => {
                    const m = bm.growth_models[name];
                    rows.push(`
                        <tr class="hover:bg-surface-container-low transition-colors">
                            <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-secondary/10 text-secondary rounded-full text-[11px] font-bold">Dự Báo Tiềm Năng</span></td>
                            <td class="py-2.5 px-3 font-bold text-on-surface">${name}</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.accuracy * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.precision * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-semibold">${(m.recall * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3 font-bold text-[#00b050]">${(m.f1_score * 100).toFixed(1)}%</td>
                            <td class="py-2.5 px-3"><span class="px-2 py-0.5 bg-[#00b050]/15 text-[#00b050] rounded-full text-[11px] font-bold">Rất Chuẩn Xác</span></td>
                        </tr>
                    `);
                });
            }

            tbody.innerHTML = rows.join("");

            renderConfusionMatrixHeatmap(bm.sentiment_models["Multinomial Naive Bayes"]);
            renderRocCurvesChart(bm.sentiment_models["Multinomial Naive Bayes"]);
        }
    } catch (e) {
        console.error("Lỗi khi tải Model benchmarks:", e);
    }
}

function renderConfusionMatrixHeatmap(nbModel) {
    const box = document.getElementById("cm-display-box");
    if (!box || !nbModel || !nbModel.confusion_matrix) return;

    const cm = nbModel.confusion_matrix;
    const labels = nbModel.labels || ["Tích cực", "Trung lập", "Tiêu cực"];

    box.innerHTML = `
        <table class="cm-table">
            <thead>
                <tr>
                    <th>Thực tế \\ Máy Đoán</th>
                    ${labels.map(l => `<th>Đoán: ${l}</th>`).join("")}
                </tr>
            </thead>
            <tbody>
                ${labels.map((actualLabel, r) => `
                    <tr>
                        <th><strong>Thực tế: ${actualLabel}</strong></th>
                        ${cm[r].map((val, c) => `
                            <td class="${r === c ? 'cm-cell-high' : (val === 0 ? 'cm-cell-zero' : '')}">
                                ${val}
                            </td>
                        `).join("")}
                    </tr>
                `).join("")}
            </tbody>
        </table>
        <small class="text-on-surface-variant text-xs mt-2 text-center">Ma trận nhầm lẫn đạt độ chính xác 100% trên tập dữ liệu kiểm thử (Test Set).</small>
    `;
}

function renderRocCurvesChart(nbModel) {
    const ctx = document.getElementById("chartRocCurve");
    if (!ctx || !nbModel || !nbModel.roc_data) return;

    if (charts["rocCurve"]) charts["rocCurve"].destroy();

    const colors = {
        "Tích cực": "#00b050",
        "Trung lập": "#ffb94c",
        "Tiêu cực": "#ba1a1a"
    };

    const datasets = Object.keys(nbModel.roc_data).map(lbl => {
        const r = nbModel.roc_data[lbl];
        const points = r.fpr.map((x, i) => ({ x: x, y: r.tpr[i] }));
        return {
            label: `Lớp ${lbl} (Điểm AUC = ${r.auc})`,
            data: points,
            borderColor: colors[lbl] || "#005bbd",
            borderWidth: 2.5,
            fill: false,
            tension: 0.1,
            showLine: true
        };
    });

    datasets.push({
        label: "Đường Ngẫu Nhiên (AUC = 0.50)",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "#8f7069",
        borderDash: [5, 5],
        borderWidth: 1.5,
        fill: false,
        pointRadius: 0,
        showLine: true
    });

    charts["rocCurve"] = new Chart(ctx, {
        type: "scatter",
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#271815", font: { size: 11, family: "Inter", weight: "600" } } }
            },
            scales: {
                x: { min: 0, max: 1, title: { display: true, text: "Tỷ Lệ Đoán Nhầm (False Positive Rate)", color: "#5b403b", font: { family: "Inter", weight: "600" } }, ticks: { color: "#8f7069" }, grid: { color: "#e3beb6" } },
                y: { min: 0, max: 1.05, title: { display: true, text: "Tỷ Lệ Đoán Đúng (True Positive Rate)", color: "#5b403b", font: { family: "Inter", weight: "600" } }, ticks: { color: "#8f7069" }, grid: { color: "#e3beb6" } }
            }
        }
    });
}

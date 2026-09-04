let charts = {};
let sampleProductsData = [];

document.addEventListener("DOMContentLoaded", () => {
    initSidebar();
    initTabs();
    if (!document.getElementById("tab-overview")) return;
    loadOverviewStats();
    loadSampleProducts();
    loadSampleComments();
    loadWordCloud("all");
    loadAprioriRules();
    loadClusteringData();
    loadModelBenchmarks();
});

function initSidebar() {
    const openButton = document.getElementById("sidebar-open");
    const closeButton = document.getElementById("sidebar-close");
    const backdrop = document.getElementById("sidebar-backdrop");
    const sidebar = document.getElementById("app-sidebar");
    if (!sidebar) return;

    const setOpen = isOpen => {
        document.body.classList.toggle("sidebar-is-open", isOpen);
        openButton?.setAttribute("aria-expanded", String(isOpen));
        backdrop?.setAttribute("aria-hidden", String(!isOpen));
        if (isOpen) closeButton?.focus();
    };

    openButton?.addEventListener("click", () => setOpen(true));
    closeButton?.addEventListener("click", () => setOpen(false));
    backdrop?.addEventListener("click", () => setOpen(false));
    sidebar.querySelectorAll("a").forEach(link => {
        link.addEventListener("click", () => {
            if (window.innerWidth < 1024) setOpen(false);
        });
    });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && document.body.classList.contains("sidebar-is-open")) {
            setOpen(false);
            openButton?.focus();
        }
    });
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    })[char]);
}

function formatNumber(value, digits = 0) {
    const number = Number(value);
    return Number.isFinite(number)
        ? number.toLocaleString("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits })
        : "—";
}

function readNumber(id, fallback) {
    const element = document.getElementById(id);
    const number = Number(element?.value);
    return Number.isFinite(number) ? number : fallback;
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function initTabs() {
    const links = document.querySelectorAll(".stitch-nav-item");
    const panes = document.querySelectorAll(".tab-pane");
    const breadcrumb = document.getElementById("stitch-breadcrumb-text");
    const names = {
        "tab-overview": "Tổng quan dữ liệu",
        "tab-sentiment": "Cảm xúc và khía cạnh",
        "tab-forecaster": "Ước lượng nhịp bán",
        "tab-risk": "Chất lượng dữ liệu",
        "tab-basket": "Sản phẩm mua cùng",
        "tab-clustering": "Phân cụm sản phẩm",
        "tab-benchmark": "Đánh giá mô hình"
    };
    const activate = target => {
        const pane = document.getElementById(target);
        if (!pane) return false;
        panes.forEach(item => item.classList.remove("active"));
        links.forEach(item => {
            item.classList.remove("active");
            item.removeAttribute("aria-current");
        });
        pane.classList.add("active");
        const link = Array.from(links).find(item => item.dataset.tab === target);
        if (link) {
            link.classList.add("active");
            link.setAttribute("aria-current", "page");
        }
        if (breadcrumb) breadcrumb.textContent = names[target] || target;
        if (window.MathJax?.typesetPromise) {
            window.MathJax.typesetPromise([pane]).catch(error => console.error(error));
        }
        return true;
    };
    links.forEach(link => link.addEventListener("click", event => {
        if (!activate(link.dataset.tab)) return;
        event.preventDefault();
        window.history.replaceState(null, "", `#${link.dataset.tab}`);
    }));
    if (window.location.hash) activate(window.location.hash.slice(1));
}

async function loadOverviewStats() {
    try {
        const response = await fetch("/api/overview_stats");
        const data = await response.json();
        if (!response.ok || data.status !== "success") throw new Error(data.message || "Không tải được trang tổng quan");
        setText("kpi-sales", data.kpis.historical_sold);
        setText("kpi-products", `${formatNumber(data.kpis.total_products)} sản phẩm`);
        setText("kpi-reviews", `${formatNumber(data.kpis.total_reviews)} đánh giá`);
        setText("kpi-rating", `${data.kpis.avg_rating} / 5 ★`);
        renderCategoryChart(data.category_sales);
        renderSentimentChart(data.sentiment_dist);
        renderRatingChart(data.rating_dist);
        renderTopProducts(data.top_products);
        renderQuality(data.quality_dist, data.listing_alerts);
    } catch (error) {
        console.error(error);
    }
}

function renderCategoryChart(source) {
    const canvas = document.getElementById("chartCategorySales");
    if (!canvas) return;
    charts.category?.destroy();
    const labels = source.labels.slice(0, 12);
    const values = source.values.slice(0, 12);
    charts.category = new Chart(canvas, {
        type: "bar",
        data: { labels, datasets: [{ label: "Lượt bán tích lũy", data: values, backgroundColor: "#ee4d2d", borderRadius: 5 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 45, minRotation: 25 } } } }
    });
}

function renderSentimentChart(source) {
    const canvas = document.getElementById("chartSentimentDoughnut");
    if (!canvas) return;
    charts.sentiment?.destroy();
    charts.sentiment = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: ["Tích cực", "Trung lập", "Tiêu cực"],
            datasets: [{ data: [source.pos, source.neu, source.neg], backgroundColor: ["#00b050", "#ffb94c", "#ba1a1a"], borderWidth: 2 }]
        },
        options: { responsive: true, maintainAspectRatio: false, cutout: "68%", plugins: { legend: { position: "bottom" } } }
    });
}

function renderRatingChart(source) {
    const canvas = document.getElementById("chartRatingDist");
    if (!canvas) return;
    charts.rating?.destroy();
    charts.rating = new Chart(canvas, {
        type: "bar",
        data: { labels: source.labels, datasets: [{ data: source.values, backgroundColor: ["#ba1a1a", "#d63c1e", "#ffb94c", "#4f92fe", "#00b050"], borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, indexAxis: "y", plugins: { legend: { display: false } } }
    });
}

function renderTopProducts(products) {
    const body = document.getElementById("top-products-tbody");
    if (!body) return;
    body.innerHTML = products.map(product => `
        <tr>
            <td class="p-2 font-mono text-primary">${escapeHtml(product.product_id)}</td>
            <td class="p-2 font-semibold"><a href="${escapeHtml(product.product_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(product.product_name.slice(0, 55))}</a></td>
            <td class="p-2">${escapeHtml(product.category)}</td>
            <td class="p-2">RM ${formatNumber(product.price, 2)}</td>
            <td class="p-2 font-bold">${formatNumber(product.historical_sold)}</td>
            <td class="p-2">${product.rating_star == null ? "—" : `★ ${formatNumber(product.rating_star, 1)}`}</td>
            <td class="p-2">${product.daily_sold_rate == null ? "Chưa đủ lần ghi nhận" : `${formatNumber(product.daily_sold_rate, 3)} sản phẩm/ngày`}</td>
        </tr>`).join("");
}

function renderQuality(quality, alerts) {
    setText("quality-count-velocity", formatNumber(quality.with_velocity));
    setText("quality-count-no-velocity", formatNumber(quality.without_velocity));
    setText("quality-count-rating", formatNumber(quality.missing_rating));
    setText("quality-count-location", formatNumber(quality.missing_location));
    const body = document.getElementById("quality-alerts-tbody");
    if (!body) return;
    body.innerHTML = alerts.map(alert => `
        <tr>
            <td class="p-2"><span class="px-2 py-1 bg-tertiary-fixed/40 text-tertiary rounded font-bold">${escapeHtml(alert.quality_level)}</span></td>
            <td class="p-2">${alert.issues.map(escapeHtml).join("; ")}</td>
            <td class="p-2"><strong>${escapeHtml(alert.product_id)}</strong><br>${escapeHtml(alert.product_name.slice(0, 70))}</td>
            <td class="p-2">RM ${formatNumber(alert.price, 2)} / ${alert.rating_star == null ? "—" : formatNumber(alert.rating_star, 1)} ★</td>
            <td class="p-2 text-on-surface-variant">${escapeHtml(alert.recommended_action)}</td>
        </tr>`).join("");
}

async function loadSampleComments() {
    try {
        const data = await (await fetch("/api/sample_comments")).json();
        const container = document.getElementById("sample-comments-container");
        if (!container || data.status !== "success") return;
        container.innerHTML = data.samples.map(sample => `
            <button class="px-3 py-1 bg-surface rounded-full border text-xs" data-comment="${encodeURIComponent(sample.text)}">${escapeHtml(sample.type)}: ${escapeHtml(sample.text.slice(0, 35))}…</button>
        `).join("");
        container.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
            document.getElementById("sentiment-input-text").value = decodeURIComponent(button.dataset.comment);
            submitSentimentAnalysis();
        }));
    } catch (error) {
        console.error(error);
    }
}

async function submitSentimentAnalysis() {
    const text = document.getElementById("sentiment-input-text")?.value.trim();
    if (!text) return;
    try {
        const response = await fetch("/api/analyze_sentiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        if (!response.ok || data.status !== "success") throw new Error(data.message || "Phân tích thất bại");
        setText("sentiment-cleaned-display", data.cleaned_text);
        const badge = document.getElementById("sent-badge-label");
        badge.textContent = data.sentiment_label;
        badge.className = `px-4 py-1.5 rounded-full font-bold ${data.sentiment_label === "Tích cực" ? "bg-green-100 text-green-700" : data.sentiment_label === "Tiêu cực" ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700"}`;
        setText("sent-score-val", `Xác suất mô hình: ${data.confidence_percent}% · Điểm từ điển: ${data.lexicon_score}`);
        document.getElementById("sent-gauge-fill").style.width = `${Math.max(0, Math.min(100, data.confidence_percent))}%`;
        renderKeywordPills("sent-pos-pills", data.positive_keywords, "bg-green-100 text-green-700");
        renderKeywordPills("sent-neg-pills", data.negative_keywords, "bg-red-100 text-red-700");
        document.getElementById("sent-aspect-pills").innerHTML = data.aspects.map(aspect => `
            <div class="bg-surface-container-low rounded-lg p-2">
                <strong>${escapeHtml(aspect.name)}</strong>: ${escapeHtml(aspect.label)}
                <span class="text-on-surface-variant">(${aspect.confidence_percent}%)</span>
            </div>`).join("");
    } catch (error) {
        console.error(error);
    }
}

function renderKeywordPills(id, words, classes) {
    const container = document.getElementById(id);
    if (!container) return;
    container.innerHTML = words?.length
        ? words.map(word => `<span class="px-2 py-0.5 rounded ${classes}">${escapeHtml(word)}</span>`).join("")
        : '<span class="text-on-surface-variant">Không phát hiện</span>';
}

async function loadWordCloud(filter, clickedButton = null) {
    document.querySelectorAll(".wc-filter-btn").forEach(button => {
        button.classList.remove("bg-primary", "text-white");
        button.classList.add("bg-surface-container");
    });
    if (clickedButton) {
        clickedButton.classList.add("bg-primary", "text-white");
        clickedButton.classList.remove("bg-surface-container");
    }
    try {
        const data = await (await fetch(`/api/wordcloud?sentiment=${encodeURIComponent(filter)}`)).json();
        const container = document.getElementById("wordcloud-container");
        if (!container || data.status !== "success") return;
        const maximum = Math.max(...data.words.map(word => word.weight), 1);
        container.innerHTML = data.words.map(word => {
            const size = 12 + Math.round(word.weight / maximum * 12);
            return `<span class="px-2 py-1 rounded bg-primary/10 text-primary" style="font-size:${size}px">${escapeHtml(word.text)} <small>${word.weight}</small></span>`;
        }).join("");
    } catch (error) {
        console.error(error);
    }
}

async function loadSampleProducts() {
    try {
        const data = await (await fetch("/api/sample_products")).json();
        if (data.status !== "success") return;
        sampleProductsData = data.samples;
        const container = document.getElementById("sample-products-container");
        if (!container) return;
        container.innerHTML = data.samples.slice(0, 6).map((product, index) => `
            <button class="px-3 py-1 bg-surface border rounded-full text-xs" onclick="populateSampleProduct(${index})">${escapeHtml(product.category)} · RM ${formatNumber(product.price, 2)}</button>
        `).join("");
    } catch (error) {
        console.error(error);
    }
}

function setSelectValue(id, value) {
    const select = document.getElementById(id);
    if (!select) return;
    const safeValue = value || "Không rõ";
    if (!Array.from(select.options).some(option => option.value === safeValue)) {
        select.add(new Option(safeValue, safeValue));
    }
    select.value = safeValue;
}

function populateSampleProduct(index) {
    const product = sampleProductsData[index];
    if (!product) return;
    populateProductFields(product);
    submitProductPrediction();
}

function populateProductFields(product) {
    document.getElementById("p-url").value = product.product_url || product.product_id;
    setSelectValue("p-category", product.category);
    setSelectValue("p-location", product.shop_location || "Không rõ");
    document.getElementById("p-price").value = product.price ?? 10.74;
    document.getElementById("p-discount").value = product.discount_rate ?? 8;
    document.getElementById("p-favs").value = product.favorite_count ?? 172;
    document.getElementById("p-hist-sold").value = product.historical_sold ?? 37;
    document.getElementById("p-rating").value = product.rating_star ?? 4.9;
    document.getElementById("p-rating-count").value = product.rating_count ?? 37;
    document.getElementById("p-snapshot-count").value = product.snapshot_count ?? 1;
}

async function loadProductReference() {
    const reference = document.getElementById("p-url").value.trim();
    const status = document.getElementById("p-lookup-status");
    if (!reference) return;
    try {
        const response = await fetch(`/api/product_lookup?reference=${encodeURIComponent(reference)}`);
        const data = await response.json();
        if (!response.ok || data.status !== "success") throw new Error(data.message || "Không tìm thấy");
        populateProductFields(data.product);
        status.textContent = `Đã nạp ${data.product.product_id} từ dữ liệu ghi nhận.`;
        status.className = "text-[11px] text-green-700 mt-1";
    } catch (error) {
        status.textContent = error.message;
        status.className = "text-[11px] text-error mt-1";
    }
}

async function submitProductPrediction() {
    const payload = {
        product_url: document.getElementById("p-url").value.trim(),
        category: document.getElementById("p-category").value,
        shop_location: document.getElementById("p-location").value,
        price: readNumber("p-price", 10.74),
        discount_rate: readNumber("p-discount", 8),
        favorite_count: readNumber("p-favs", 172),
        historical_sold: readNumber("p-hist-sold", 37),
        rating_star: readNumber("p-rating", 4.9),
        rating_count: readNumber("p-rating-count", 37),
        snapshot_count: readNumber("p-snapshot-count", 2)
    };
    try {
        const response = await fetch("/api/predict_product", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || data.status !== "success") throw new Error(data.message || "Ước lượng thất bại");
        const prediction = data.prediction;
        document.getElementById("res-placeholder").classList.add("hidden");
        document.getElementById("res-actual-content").classList.remove("hidden");
        setText("res-performance-band", prediction.performance_band);
        setText("res-daily-rate", formatNumber(prediction.estimated_daily_sold_rate, 3));
        setText("res-30d-rate", `${formatNumber(prediction.estimated_30d_sales_pace, 1)} sản phẩm`);
        setText("res-30d-value", `RM ${formatNumber(prediction.estimated_30d_gross_value_myr, 2)}`);
        setText("res-method-note", prediction.method_note);
        document.getElementById("res-feature-bars").innerHTML = prediction.feature_impacts.map(feature => `
            <div class="grid grid-cols-[130px_1fr_55px] gap-2 items-center text-xs">
                <span>${escapeHtml(feature.name)}</span><div class="h-2 bg-surface-container rounded-full overflow-hidden"><div class="h-full bg-primary" style="width:${feature.score}%"></div></div><span>${feature.importance_percent}%</span>
            </div>`).join("");
    } catch (error) {
        const status = document.getElementById("p-lookup-status");
        status.textContent = error.message;
        status.className = "text-[11px] text-error mt-1";
    }
}

function friendlyBasketName(rawName) {
    const name = String(rawName ?? "").replace(/\s+/g, " ").trim();
    if (/^sterofoam extra$/i.test(name)) return "Hộp xốp bổ sung";
    if (!/pempek/i.test(name)) return name;

    let base = /tenggiri/i.test(name) ? "Pempek cá thu AlKindi" : "Pempek Palembang AlKindi";
    if (/steril/i.test(name)) base += " tiệt trùng";
    const details = [];
    const count = name.match(/(\d+)\s*pcs/i);
    if (count) details.push(`gói ${count[1]} cái`);
    const size = name.match(/size\s*(mini|medium)/i);
    if (size) details.push(size[1].toLowerCase() === "mini" ? "cỡ mini" : "cỡ vừa");
    const variants = [];
    if (/lenjer/i.test(name)) variants.push("thanh");
    if (/kulit/i.test(name)) variants.push("da cá");
    if (/telur/i.test(name)) variants.push("trứng");
    if (/adaan/i.test(name)) variants.push("viên chiên");
    if (variants.length) details.push(`loại ${variants.join("/")}`);
    return details.length ? `${base} — ${details.join(", ")}` : base;
}

function basketProductCell(rawName) {
    return `
        <strong>${escapeHtml(friendlyBasketName(rawName))}</strong>
        <details class="mt-1 text-[10px] text-on-surface-variant">
            <summary class="cursor-pointer">Xem tên gốc trong dữ liệu</summary>
            <span>${escapeHtml(rawName)}</span>
        </details>`;
}

async function loadAprioriRules() {
    try {
        const data = await (await fetch("/api/market_basket")).json();
        const body = document.getElementById("apriori-rules-tbody");
        if (!body || data.status !== "success") return;
        if (!data.rules.length) {
            body.innerHTML = '<tr><td colspan="6" class="p-4 text-center">Không có cặp sản phẩm đạt ngưỡng.</td></tr>';
            return;
        }

        // Tương thích cả với tiến trình Flask đang giữ JSON cũ trong RAM: suy ra các
        // trường diễn giải từ support/confidence nếu server chưa được khởi động lại.
        const rules = data.rules.map(rule => ({
            ...rule,
            antecedent_orders: rule.antecedent_orders
                ?? Math.round(rule.cooccurrence_orders / (rule.confidence / 100)),
            actionable: rule.actionable ?? rule.lift > 1,
            association_level: rule.association_level
                ?? (rule.lift >= 2 ? "Liên hệ mạnh" : rule.lift > 1 ? "Có xu hướng đi kèm" : "Không nên gợi ý mua cùng")
        }));
        const strongest = rules.find(rule => rule.actionable) || rules[0];
        const example = document.getElementById("apriori-example");
        if (example) {
            example.innerHTML = `
                <strong class="text-on-surface">Ví dụ đọc dòng đầu:</strong>
                Trong <strong>${strongest.antecedent_orders} đơn</strong> có “${escapeHtml(friendlyBasketName(strongest.antecedent))}”,
                có <strong>${strongest.cooccurrence_orders} đơn</strong> mua thêm “${escapeHtml(friendlyBasketName(strongest.consequent))}”
                — tức <strong>${strongest.confidence}%</strong>. Hai món cùng xuất hiện trong
                <strong>${strongest.cooccurrence_orders}/${strongest.completed_order_count} đơn (${strongest.support}%)</strong>;
                Lift <strong>${strongest.lift}</strong> nghĩa là mức đi kèm cao gấp khoảng ${strongest.lift} lần so với khi xem hai món độc lập.`;
        }

        body.innerHTML = rules.map(rule => {
            const useful = rule.actionable;
            const badgeClass = rule.lift >= 2
                ? "bg-green-100 text-green-800"
                : useful
                    ? "bg-blue-100 text-blue-800"
                    : "bg-red-100 text-red-800";
            const interpretation = useful
                ? `${rule.cooccurrence_orders}/${rule.antecedent_orders} đơn có A cũng có B; mức đi kèm cao hơn kỳ vọng độc lập.`
                : `${rule.cooccurrence_orders}/${rule.antecedent_orders} đơn có A cũng có B, nhưng lift dưới 1 nên không dùng để gợi ý mua cùng.`;
            const associationLabel = useful
                ? rule.association_level
                : "Không nên gợi ý mua cùng";
            return `
                <tr class="${useful ? "" : "bg-surface-container-low/60"}">
                    <td class="p-3 align-top">${basketProductCell(rule.antecedent)}</td>
                    <td class="p-3 align-top">${basketProductCell(rule.consequent)}</td>
                    <td class="p-3 align-top"><strong>${rule.cooccurrence_orders}/${rule.completed_order_count} đơn</strong><br><span class="text-on-surface-variant">${rule.support}% tổng số đơn</span></td>
                    <td class="p-3 align-top"><strong>${rule.cooccurrence_orders}/${rule.antecedent_orders} đơn</strong><br><span class="text-on-surface-variant">${rule.confidence}% đơn có A</span></td>
                    <td class="p-3 align-top"><span class="inline-block px-2 py-1 rounded font-bold ${badgeClass}">${escapeHtml(associationLabel)}</span><br><span class="inline-block mt-1 font-semibold">Lift ${rule.lift}</span></td>
                    <td class="p-3 align-top text-on-surface-variant">${escapeHtml(interpretation)}</td>
                </tr>`;
        }).join("");
    } catch (error) {
        console.error(error);
    }
}

async function loadClusteringData() {
    try {
        const data = await (await fetch("/api/clustering")).json();
        if (data.status !== "success") return;
        const clustering = data.clustering;
        const container = document.getElementById("cluster-profiles-container");
        if (container) {
            container.innerHTML = Object.values(clustering.cluster_profiles).map(profile => `
                <div class="bg-surface-container-low rounded-lg p-3 text-xs">
                    <strong class="text-primary">${escapeHtml(profile.name)}</strong>
                    <p class="mt-1">${formatNumber(profile.count)} sản phẩm · Giá trung bình RM ${formatNumber(profile.avg_price, 2)}</p>
                    <p>Đã bán trung bình ${formatNumber(profile.avg_historical_sold, 1)} · Có nhịp bán ${(profile.velocity_coverage * 100).toFixed(1)}%</p>
                </div>`).join("");
        }
        renderClusterChart(clustering.sample_points);
    } catch (error) {
        console.error(error);
    }
}

function renderClusterChart(points) {
    const canvas = document.getElementById("chartKmeansScatter");
    if (!canvas) return;
    charts.cluster?.destroy();
    const colors = ["#ee4d2d", "#1769aa", "#039855", "#f79009"];
    const datasets = [0, 1, 2, 3].map(cluster => {
        const members = points.filter(point => point.cluster === cluster);
        return {
            label: members[0]?.cluster_name || `Cụm ${cluster}`,
            data: members.map(point => ({ x: point.pca_x, y: point.pca_y, ...point })),
            backgroundColor: colors[cluster], pointRadius: 5, pointHoverRadius: 8
        };
    });
    charts.cluster = new Chart(canvas, {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: context => `${context.raw.product_name} · RM ${formatNumber(context.raw.price, 2)} · đã bán ${formatNumber(context.raw.historical_sold)}` } } },
            scales: { x: { title: { display: true, text: "PCA 1" } }, y: { title: { display: true, text: "PCA 2" } } }
        }
    });
}

function aspectDisplayName(key) {
    const names = {
        price_sentiment: "Giá",
        shipping_sentiment: "Giao hàng",
        outlook_sentiment: "Hình thức",
        quality_sentiment: "Chất lượng",
        size_sentiment: "Kích cỡ",
        shop_service_sentiment: "Dịch vụ của shop",
        general_sentiment: "Trải nghiệm chung",
        others_sentiment: "Khía cạnh khác"
    };
    return names[key] || key;
}

async function loadModelBenchmarks() {
    try {
        const data = await (await fetch("/api/model_benchmarks")).json();
        if (data.status !== "success") return;
        const benchmark = data.benchmarks;
        const rows = [];
        Object.entries(benchmark.sentiment_models || {}).forEach(([name, metric]) => rows.push(`
            <tr><td class="p-2">Cảm xúc tổng quát</td><td class="p-2 font-bold">${escapeHtml(name)}</td><td class="p-2">${(metric.accuracy * 100).toFixed(2)}%</td><td class="p-2">${(metric.macro_f1 * 100).toFixed(2)}%</td><td class="p-2">${(metric.f1_score * 100).toFixed(2)}%</td><td class="p-2">${metric.test_samples}</td></tr>
        `));
        Object.entries(benchmark.aspect_models || {}).forEach(([name, metric]) => rows.push(`
            <tr><td class="p-2">Khía cạnh: ${escapeHtml(aspectDisplayName(name))}</td><td class="p-2 font-bold">Logistic Regression</td><td class="p-2">${(metric.accuracy * 100).toFixed(2)}%</td><td class="p-2">${(metric.macro_f1 * 100).toFixed(2)}%</td><td class="p-2">${(metric.f1_score * 100).toFixed(2)}%</td><td class="p-2">${metric.test_samples}</td></tr>
        `));
        const sales = benchmark.sales_velocity_regressor;
        Object.entries(sales.models || {}).forEach(([name, metric]) => rows.push(`
            <tr><td class="p-2">Nhịp bán quan sát</td><td class="p-2 font-bold">${escapeHtml(name)}</td><td class="p-2">R² ${metric.r2}</td><td class="p-2">MAE ${metric.mae}</td><td class="p-2">RMSE ${metric.rmse}</td><td class="p-2">${sales.test_samples}</td></tr>
        `));
        document.getElementById("benchmark-tbody").innerHTML = rows.join("");
        const selected = benchmark.sentiment_models[benchmark.selected_sentiment_model];
        renderConfusionMatrix(selected);
        renderRoc(selected);
    } catch (error) {
        console.error(error);
    }
}

function renderConfusionMatrix(metric) {
    const container = document.getElementById("cm-display-box");
    if (!container || !metric) return;
    container.innerHTML = `<table class="w-full text-xs text-center"><thead><tr><th class="p-2">Thực tế \\ Dự đoán</th>${metric.labels.map(label => `<th class="p-2">${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>${metric.labels.map((label, row) => `<tr><th class="p-2">${escapeHtml(label)}</th>${metric.confusion_matrix[row].map((value, column) => `<td class="p-2 ${row === column ? "bg-green-100 font-bold" : "bg-surface-container-low"}">${value}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function renderRoc(metric) {
    const canvas = document.getElementById("chartRocCurve");
    if (!canvas || !metric) return;
    charts.roc?.destroy();
    const colors = { "Tích cực": "#039855", "Trung lập": "#f79009", "Tiêu cực": "#b42318" };
    const datasets = Object.entries(metric.roc_data || {}).map(([label, values]) => ({
        label: `${label} (AUC ${values.auc})`,
        data: values.fpr.map((x, index) => ({ x, y: values.tpr[index] })),
        borderColor: colors[label] || "#005bbd", pointRadius: 1, fill: false
    }));
    charts.roc = new Chart(canvas, {
        type: "line", data: { datasets },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Tỷ lệ dương tính giả" } }, y: { min: 0, max: 1, title: { display: true, text: "Tỷ lệ dương tính đúng" } } } }
    });
}

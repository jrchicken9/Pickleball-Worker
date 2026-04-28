create table if not exists public.training_status (
  run_id text primary key,
  state text not null default 'running',
  epoch integer not null default 0,
  epochs integer not null default 0,
  batch integer not null default 0,
  batches_per_epoch integer not null default 0,
  epoch_progress double precision not null default 0,
  overall_progress double precision not null default 0,
  elapsed_seconds double precision not null default 0,
  save_dir text not null default '',
  updated_at_unix double precision not null default 0
);

alter table public.training_status enable row level security;

drop policy if exists "public read training_status" on public.training_status;
drop policy if exists "public insert training_status" on public.training_status;
drop policy if exists "public update training_status" on public.training_status;

create policy "public read training_status"
on public.training_status for select to anon using (true);

create policy "public insert training_status"
on public.training_status for insert to anon with check (true);

create policy "public update training_status"
on public.training_status for update to anon using (true) with check (true);
